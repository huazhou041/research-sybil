"""
TANGERINE-exact 3D CT preprocessing.

Reproduces the official TANGERINE foundation-model preprocessing
(3D-MAE-MedImaging/datasets_three_d_fine.py) while reusing the robust
DICOM-loading framework from the Swin preprocessing pipeline.

Official TANGERINE spec (ground truth from the repo):
    1. Resample to FIXED size 256x256x256 with DYNAMIC spacing
       new_spacing = (original_size * original_spacing) / 256
       (preserves each scan's physical extent — NO cropping)
    2. BSpline interpolation
    3. HU clip to [-1200, 800]   (applied at load time in the original;
       here we save RAW resampled HU and let the dataset clip+normalize,
       matching Custom3DDataset exactly)
    4. Fixed min-max normalization (volume + 1200) / 2000  -> [0, 1]
       (done in the dataset loader, NOT here)

Differences from the Swin preprocessor this is based on:
    * intensity_range [-1200, 800]  (Swin used [-1000, 400])
    * resample to 256^3 dynamic spacing (Swin used 1mm isotropic)
    * BSpline interpolation (Swin used linear)
    * NO Swin window/patch padding (TANGERINE wants exactly 256^3)
    * Saves RAW resampled HU so the dataset can apply the official
      fixed normalization at load time.

Robust DICOM handling (inherited from Swin pipeline):
    * series reader with slice-by-slice fallback on inconsistency
    * size filtering (keep most common slice size)
    * rescale slope/intercept applied
    * resumable batch processing via per-job JSON logs

Usage (single job):
    python preprocess_tangerine3d.py \
        --data_root  /path/to/dicom_root \
        --output_root /path/to/images_3d_tangerine \
        --job_id 0 --total_jobs 1

Usage (SLURM array): see batch_preprocess_tangerine.sh
"""
import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import SimpleITK as sitk

from series_selection import select_best_series


def _atomic_json_dump(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(rows, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _validate_nifti(path, expected_shape=None):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise IOError(f'Output file missing or empty: {path}')
    img = sitk.ReadImage(str(path))
    if expected_shape is not None:
        got = tuple(reversed(img.GetSize()))
        exp = tuple(expected_shape)
        if got != exp:
            raise IOError(f'Output shape mismatch for {path}: got {got}, expected {exp}')
    return True


def _valid_existing_output(path, output_format, expected_shape=None):
    path = Path(path)
    try:
        if output_format == 'nifti':
            _validate_nifti(path, expected_shape)
        else:
            if not path.exists() or path.stat().st_size == 0:
                return False
            with np.load(path) as z:
                arr = z['image']
                if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
                    return False
        return True
    except Exception:
        return False



class TangerineExact3DPreprocessor:
    def __init__(self,
                 target_size=(256, 256, 256),
                 intensity_range=(-1200, 800),
                 interpolator='bspline',
                 output_format='nifti'):
        """
        Args:
            target_size: fixed output size (D, H, W) — TANGERINE uses 256^3
            intensity_range: HU clip range — TANGERINE uses (-1200, 800).
                             Used as the resample default (padding) value.
                             NOTE: we do NOT normalize here; the dataset loader
                             applies the official fixed normalization.
            interpolator: 'bspline' (TANGERINE) or 'linear'
            output_format: 'nifti' (.nii.gz) or 'npz'
        """
        self.target_size = tuple(target_size)
        self.intensity_range = tuple(intensity_range)
        self.interpolator = interpolator
        self.output_format = output_format

    # ── DICOM loading (robust, from Swin pipeline) ─────────────────────────
    def load_dicom_series(self, dicom_folder):
        # Select ONE coherent series (axial, thin, largest) — NOT the arbitrary
        # first series. Folders often contain 8+ mixed series (thin/thick axial,
        # sag/cor reformats, scouts); the default reader would grab the wrong one.
        dicom_names, sel_info = select_best_series(dicom_folder)
        if len(dicom_names) == 0:
            raise ValueError(f"No usable DICOM series in {dicom_folder}")

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_names)
        try:
            image = reader.Execute()
        except RuntimeError as e:
            error_msg = str(e)
            if ("ImageIO returns IO region" in error_msg or
                "Inputs do not occupy the same physical space" in error_msg or
                "Size mismatch" in error_msg or
                "does not match the required size" in error_msg):
                print(f"[WARNING] DICOM inconsistency detected: {dicom_folder}")
                print(f"[INFO] Using manual reconstruction with size filtering...")
                image = self._reconstruct_from_slices(dicom_names, dicom_folder)
            else:
                raise

        metadata = self._extract_metadata(dicom_names, image)
        metadata['series_selection'] = sel_info.get('selection')
        metadata['series_orientation'] = sel_info.get('orientation')
        metadata['series_n_total'] = sel_info.get('n_series_total')
        metadata['series_desc'] = sel_info.get('desc')
        return image, metadata

    def _reconstruct_from_slices(self, dicom_names, dicom_folder):
        slices_data = []
        for dcm_file in dicom_names:
            try:
                sr = sitk.ImageFileReader()
                sr.SetFileName(dcm_file)
                simg = sr.Execute()
                slices_data.append({
                    'image': simg,
                    'z_position': simg.GetOrigin()[2],
                    'size': simg.GetSize(),
                    'file': dcm_file,
                })
            except Exception as slice_err:
                print(f"[WARNING] Skipping corrupted slice {dcm_file}: {slice_err}")
                continue

        if len(slices_data) == 0:
            raise ValueError(f"No valid slices in {dicom_folder}")

        # Keep most common slice size
        sizes = [s['size'] for s in slices_data]
        size_counts = Counter(sizes)
        print(f"[INFO] Size variants found: {dict(size_counts)}")
        target_size = size_counts.most_common(1)[0][0]
        original_count = len(slices_data)
        slices_data = [s for s in slices_data if s['size'] == target_size]
        if len(slices_data) < original_count:
            print(f"[WARNING] Removed {original_count - len(slices_data)} mismatched-size slices "
                  f"(kept {len(slices_data)}/{original_count})")
        if len(slices_data) < 10:
            raise ValueError(f"Too few valid slices ({len(slices_data)}) after filtering")

        # Sort by z, compute median z-spacing
        slices_data.sort(key=lambda x: x['z_position'])
        z_positions = [s['z_position'] for s in slices_data]
        if len(z_positions) > 1:
            median_z_spacing = float(np.median(np.abs(np.diff(z_positions))))
            if median_z_spacing <= 0:
                median_z_spacing = 1.0
        else:
            median_z_spacing = 1.0

        ref = slices_data[0]['image']
        xy_spacing = ref.GetSpacing()[:2]
        origin = list(ref.GetOrigin())
        direction = ref.GetDirection()

        # Stack arrays (verify consistent in-plane shape)
        slice_arrays = [sitk.GetArrayFromImage(s['image']) for s in slices_data]
        shapes = [a.shape for a in slice_arrays]
        common_shape = Counter(shapes).most_common(1)[0][0]
        keep = [a for a in slice_arrays if a.shape == common_shape]
        # Each slice array is (1, H, W) or (H, W); squeeze to 2D then stack on z
        vol = np.stack([a.squeeze() for a in keep], axis=0)  # (Z, H, W)

        out = sitk.GetImageFromArray(vol.astype(np.float32))
        out.SetSpacing((float(xy_spacing[0]), float(xy_spacing[1]), median_z_spacing))
        out.SetOrigin(origin)
        # 3D direction from 2D ref if needed
        if len(direction) == 4:
            d = [direction[0], direction[1], 0.0,
                 direction[2], direction[3], 0.0,
                 0.0, 0.0, 1.0]
            out.SetDirection(d)
        return out

    def _extract_metadata(self, dicom_names, image):
        reader_meta = sitk.ImageSeriesReader()
        reader_meta.SetFileNames([dicom_names[0]])
        reader_meta.MetaDataDictionaryArrayUpdateOn()
        reader_meta.LoadPrivateTagsOn()
        try:
            reader_meta.Execute()
        except Exception:
            print(f"[WARNING] Could not read metadata from {dicom_names[0]}")

        metadata = {}
        tags = {'0010|0010': 'PatientName', '0010|0020': 'PatientID',
                '0010|0030': 'PatientBirthDate', '0010|0040': 'PatientSex',
                '0010|1010': 'PatientAge', '0008|0020': 'StudyDate',
                '0008|0030': 'StudyTime', '0008|103e': 'SeriesDescription',
                '0020|000e': 'SeriesInstanceUID', '0008|0060': 'Modality',
                '0008|0070': 'Manufacturer', '0008|1090': 'ManufacturerModelName',
                '0018|0050': 'SliceThickness', '0018|0088': 'SpacingBetweenSlices',
                '0028|0030': 'PixelSpacing', '0028|1050': 'WindowCenter',
                '0028|1051': 'WindowWidth', '0028|1052': 'RescaleIntercept',
                '0028|1053': 'RescaleSlope'}
        for tag, name in tags.items():
            try:
                metadata[name] = reader_meta.GetMetaData(0, tag) if reader_meta.HasMetaDataKey(0, tag) else ''
            except Exception:
                metadata[name] = ''

        metadata['NumberOfSlices'] = len(dicom_names)
        sp = image.GetSpacing()
        metadata['OriginalSpacingX'] = float(sp[0])
        metadata['OriginalSpacingY'] = float(sp[1])
        metadata['OriginalSpacingZ'] = float(sp[2])
        sz = image.GetSize()
        metadata['ActualImageSizeX'] = int(sz[0])
        metadata['ActualImageSizeY'] = int(sz[1])
        metadata['ActualImageSizeZ'] = int(sz[2])
        return metadata

    # ── Resample to fixed 256^3 with dynamic spacing (TANGERINE-exact) ──────
    def resample_to_fixed_size(self, image):
        """
        Resample to EXACTLY target_size with dynamic spacing to preserve
        physical dimensions:  new_spacing = (orig_size * orig_spacing) / new_size
        Matches 3D-MAE-MedImaging/datasets_three_d_fine.py:resample_ct_volume.
        """
        original_size = np.array(image.GetSize(), dtype=np.float64)      # (W, H, D) in sitk order
        original_spacing = np.array(image.GetSpacing(), dtype=np.float64)
        # sitk GetSize/GetSpacing are (x, y, z); target_size is given (D,H,W) conceptually
        # but we keep sitk's (x,y,z) ordering consistently.
        target = np.array(self.target_size, dtype=np.float64)

        new_spacing = (original_size * original_spacing) / target

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([int(s) for s in self.target_size])
        resampler.SetOutputSpacing([float(s) for s in new_spacing])
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        if self.interpolator == 'bspline':
            resampler.SetInterpolator(sitk.sitkBSpline)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
        # Pad with HU minimum (air) — matches TANGERINE default pixel handling
        resampler.SetDefaultPixelValue(float(self.intensity_range[0]))
        return resampler.Execute(image)

    # ── Single folder ──────────────────────────────────────────────────────
    def preprocess_single_dicom_folder(self, dicom_folder, output_path):
        try:
            image, metadata = self.load_dicom_series(dicom_folder)
            arr = sitk.GetArrayFromImage(image)

            # SimpleITK/GDCM already applies CT RescaleSlope/RescaleIntercept
            # when reading DICOM into physical pixel values. Applying it again
            # would shift HU values, commonly by another -1024.
            rescale_applied = False
            resampled = self.resample_to_fixed_size(image)
            resampled_arr = sitk.GetArrayFromImage(resampled)

            # Save RAW resampled HU (the dataset loader applies clip + fixed norm,
            # exactly like the official Custom3DDataset). We do NOT normalize here.
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_path = Path(output_path)
            if self.output_format == 'nifti':
                out_img = sitk.GetImageFromArray(resampled_arr.astype(np.float32))
                out_img.SetSpacing(resampled.GetSpacing())
                out_img.SetOrigin(resampled.GetOrigin())
                out_img.SetDirection(resampled.GetDirection())
                fd, tmp_name = tempfile.mkstemp(prefix=output_path.name + '.', suffix='.tmp.nii.gz', dir=str(output_path.parent))
                os.close(fd)
                os.unlink(tmp_name)
                try:
                    sitk.WriteImage(out_img, tmp_name)
                    _validate_nifti(tmp_name, expected_shape=resampled_arr.shape)
                    os.replace(tmp_name, output_path)
                    _validate_nifti(output_path, expected_shape=resampled_arr.shape)
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
            else:
                fd, tmp_name = tempfile.mkstemp(prefix=output_path.name + '.', suffix='.tmp.npz', dir=str(output_path.parent))
                os.close(fd)
                try:
                    np.savez_compressed(tmp_name, image=resampled_arr.astype(np.float32))
                    if os.path.getsize(tmp_name) == 0:
                        raise IOError(f'Output file missing or empty: {tmp_name}')
                    os.replace(tmp_name, output_path)
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)

            result = {
                'success': True,
                'original_shape': list(arr.shape),
                'final_shape': list(resampled_arr.shape),
                'original_spacing': list(image.GetSpacing()),
                'new_spacing': list(resampled.GetSpacing()),
                'rescale_applied': rescale_applied,
                'interpolator': self.interpolator,
                'intensity_range_clip': list(self.intensity_range),
                'hu_min_raw': float(resampled_arr.min()),
                'hu_max_raw': float(resampled_arr.max()),
                'target_size': list(self.target_size),
            }
            result.update(metadata)
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ── Scan / batch (resumable) ───────────────────────────────────────────
    def scan_and_organize_dicoms(self, data_root):
        data_path = Path(data_root)
        dicom_folders = []
        try:
            with os.scandir(data_path) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=True):
                        folder = Path(entry.path)
                        if len(list(folder.glob('*.dcm'))) > 0:
                            dicom_folders.append(folder)
        except Exception as e:
            print(f"[ERROR] Failed to scan {data_path}: {e}")
            return []
        print(f"[INFO] Found {len(dicom_folders)} folders with DICOM files")
        return sorted(dicom_folders, key=lambda p: p.name)

    def get_completed_scans(self, output_root):
        output_root = Path(output_root)
        completed = set()
        ext = 'nii.gz' if self.output_format == 'nifti' else 'npz'
        expected = tuple(reversed(self.target_size))
        for log_file in [output_root / 'preprocessing_log.json', *sorted(output_root.glob('preprocessing_log_job_*.json'))]:
            if log_file.exists():
                try:
                    with open(log_file) as f:
                        for r in json.load(f):
                            name = r.get('folder_name', '')
                            if r.get('success') and name:
                                out_file = output_root / f'{name}.{ext}'
                                if _valid_existing_output(out_file, self.output_format, expected):
                                    completed.add(name)
                except Exception:
                    pass
        for out_file in output_root.glob(f'*.{ext}'):
            name = out_file.name[:-len(f'.{ext}')]
            if _valid_existing_output(out_file, self.output_format, expected):
                completed.add(name)
        return completed

    def batch_preprocess(self, data_root, output_root, job_id=0, total_jobs=1, resume=True):
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)

        completed = self.get_completed_scans(output_root) if resume else set()
        if completed:
            print(f"[INFO] Found {len(completed)} completed scans (resume mode)")

        folders = self.scan_and_organize_dicoms(data_root)
        if len(folders) == 0:
            print(f"[ERROR] No DICOM folders found! Check: {data_root}")
            return []

        all_scans = [{'folder': f, 'folder_name': f.name} for f in folders]
        per_job = len(all_scans) // total_jobs
        start = job_id * per_job
        end = start + per_job if job_id < total_jobs - 1 else len(all_scans)
        job_scans = all_scans[start:end]

        print(f"\n[JOB {job_id}/{total_jobs}] scans {start}..{end-1}  (n={len(job_scans)})")
        print(f"[INFO] TANGERINE-exact: {self.target_size} dynamic spacing, "
              f"clip {self.intensity_range}, {self.interpolator}")
        print("=" * 60)

        results, skipped = [], 0
        for idx, scan in enumerate(job_scans):
            if resume and scan['folder_name'] in completed:
                skipped += 1
                continue
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"[PROGRESS] Job {job_id}: {idx+1}/{len(job_scans)} ({skipped} skipped)")
                sys.stdout.flush()

            ext = 'nii.gz' if self.output_format == 'nifti' else 'npz'
            out_file = output_path / f"{scan['folder_name']}.{ext}"
            r = self.preprocess_single_dicom_folder(scan['folder'], out_file)
            r.update({'folder_name': scan['folder_name'], 'output_file': str(out_file), 'job_id': job_id})
            results.append(r)
            _atomic_json_dump(results, output_path / f'preprocessing_log_job_{job_id}.json')

        job_log = output_path / f'preprocessing_log_job_{job_id}.json'
        _atomic_json_dump(results, job_log)

        ok = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}\n[JOB {job_id}] done. processed={len(results)} success={ok} "
              f"failed={len(results)-ok} skipped={skipped}\n{'='*60}")
        return results


def merge_job_logs(output_root, total_jobs):
    output_path = Path(output_root)
    all_results = []
    for job_id in range(total_jobs):
        jl = output_path / f'preprocessing_log_job_{job_id}.json'
        if jl.exists():
            with open(jl) as f:
                all_results.extend(json.load(f))
    if not all_results:
        print("[WARNING] No logs to merge")
        return
    _atomic_json_dump(all_results, output_path / 'preprocessing_log.json')
    pd.DataFrame(all_results).to_csv(output_path / 'preprocessing_log.csv', index=False)
    ok = sum(1 for r in all_results if r['success'])
    with open(output_path / 'preprocessing_summary.txt', 'w') as f:
        f.write(f"TANGERINE-exact 3D Preprocessing Summary\n{'='*60}\n")
        f.write(f"Total: {len(all_results)}\nSuccess: {ok}\nFailed: {len(all_results)-ok}\n")
        f.write(f"Target: 256^3 dynamic spacing, BSpline\nClip: [-1200, 800] (norm at load)\n")
        f.write(f"Jobs: {total_jobs}\nTime: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    print(f"[MERGE] {len(all_results)} scans, {ok} success")


def main():
    p = argparse.ArgumentParser(description='TANGERINE-exact 3D CT preprocessing')
    p.add_argument('--data_root')
    p.add_argument('--output_root', required=True)
    p.add_argument('--job_id', type=int, default=0)
    p.add_argument('--total_jobs', type=int, default=1)
    p.add_argument('--target_size', type=int, nargs=3, default=[256, 256, 256])
    p.add_argument('--intensity_min', type=int, default=-1200)
    p.add_argument('--intensity_max', type=int, default=800)
    p.add_argument('--interpolator', choices=['bspline', 'linear'], default='bspline')
    p.add_argument('--format', choices=['nifti', 'npz'], default='nifti')
    p.add_argument('--no_resume', action='store_true')
    p.add_argument('--merge_logs', action='store_true')
    args = p.parse_args()

    if not args.merge_logs and not args.data_root:
        p.error('--data_root is required unless --merge_logs is used')

    if args.merge_logs:
        merge_job_logs(args.output_root, args.total_jobs)
        return

    pre = TangerineExact3DPreprocessor(
        target_size=tuple(args.target_size),
        intensity_range=(args.intensity_min, args.intensity_max),
        interpolator=args.interpolator,
        output_format=args.format,
    )
    pre.batch_preprocess(args.data_root, args.output_root,
                         args.job_id, args.total_jobs, not args.no_resume)


if __name__ == '__main__':
    main()
