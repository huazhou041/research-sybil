"""
Sybil-exact 3D CT preprocessing — saves ONE file per CT for retraining/ensembling.

Replicates the official Sybil `Serie` preprocessing pipeline so the volumes are
geometrically identical to what Sybil's model consumes, but saves a compact
SINGLE-CHANNEL, UN-NORMALIZED volume (z-score norm + 3-channel expansion are
applied at load time by the dataset). This keeps storage ~3x smaller while
remaining byte-faithful after the load-time transform.

Sybil pipeline reproduced (from sybil/serie.py + sybil/loaders + augmentations):
  1. Per-SLICE thickness filter: keep slices with 0 < SliceThickness <= 2.5mm
       (a whole CT is dropped only if NO slice survives — matches run_pred.py)
  2. Per slice: apply_modality_lut -> HU window (center=-600, width=1500) -> //256
  3. Per slice: resize to 256x256 (bilinear)
  4. Order slices by ImagePositionPatient z
  5. Stack -> resample to voxel spacing (0.703125, 0.703125, 2.5) [torchio]
  6. CropOrPad to (256, 256, 200), pad value 0
  7. Save 1-channel volume as .nii.gz

At LOAD time (dataset), apply Sybil normalization to reproduce the exact input:
    img = (vol - 128.1722) / 87.1849   ; then expand to 3 channels

Output: <output_root>/<ct_id>.nii.gz   (intended: .../images_3d_tan_syb/sybil/)

Usage (SLURM array): see batch_preprocess_sybil.sh
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
import pydicom
import cv2
import torch
import torchio as tio
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


# ── Sybil constants (must match sybil package exactly) ─────────────────────
WINDOW_CENTER = -600
WINDOW_WIDTH = 1500
VOXEL_SPACING = (0.703125, 0.703125, 2.5)   # sybil/datasets/utils.py
IMG_SIZE = (256, 256)                        # sybil args.img_size
NUM_IMAGES = 200                             # sybil args.num_images
MAX_SLICE_THICKNESS = 2.5                    # per-slice filter (run_pred.py)
# Sybil normalization (applied at LOAD time, recorded here for reference)
SYBIL_IMG_MEAN = 128.1722
SYBIL_IMG_STD = 87.1849


def apply_modality_lut(pixel_array, ds):
    """Apply RescaleSlope/RescaleIntercept to convert stored values to HU.
    Manual implementation — avoids pydicom pixel_data_handlers which trigger
    the Pillow decoder bug on some cluster environments."""
    slope     = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    arr = pixel_array.astype(np.float32)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept
    return arr


def apply_windowing(image, center, width, bit_size=16):
    """DICOM windowing — exact copy of sybil/loaders/image_loaders.py."""
    y_min = 0
    y_max = 2 ** bit_size - 1
    y_range = y_max - y_min
    c = center - 0.5
    w = width - 1
    below = image <= (c - w / 2)
    above = image > (c + w / 2)
    between = np.logical_and(~below, ~above)
    image[below] = y_min
    image[above] = y_max
    if between.any():
        image[between] = ((image[between] - c) / w + 0.5) * y_range + y_min
    return image


class SybilExact3DPreprocessor:
    def __init__(self,
                 max_slice_thickness=MAX_SLICE_THICKNESS,
                 voxel_spacing=VOXEL_SPACING,
                 img_size=IMG_SIZE,
                 num_images=NUM_IMAGES,
                 output_format='nifti'):
        self.max_thick = max_slice_thickness
        self.voxel_spacing = tuple(voxel_spacing)
        self.img_size = tuple(img_size)
        self.num_images = num_images
        self.output_format = output_format
        self.resample_transform = tio.transforms.Resample(target=self.voxel_spacing)
        self.padding_transform = tio.transforms.CropOrPad(
            target_shape=tuple(list(self.img_size) + [self.num_images]),
            padding_mode=0)

    # ── series selection + per-slice safety filter ─────────────────────────
    def filter_slices(self, dicom_folder):
        """Select ONE coherent series (axial, thin, largest), then return its
        (path, z) slices. This replaces the old glob-all-and-filter approach,
        which mixed axial + sagittal + coronal reformats into one volume.
        A light per-slice thickness check is kept as a safety net.
        Returns (kept, pixel_spacing, patient_id, sel_info)."""
        files, sel_info = select_best_series(
            dicom_folder, min_slices=10, prefer_thick_max=self.max_thick)
        kept = []
        pixel_spacing = None
        patient_id = None
        for p in files:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                thick = getattr(ds, 'SliceThickness', None)
                thick = float(thick) if thick is not None else None
                if thick is not None and (thick <= 0 or thick > self.max_thick + 1e-6):
                    continue
                z = float(ds.ImagePositionPatient[-1])
                kept.append((str(p), z, thick))
                if pixel_spacing is None and hasattr(ds, 'PixelSpacing'):
                    pixel_spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
                if patient_id is None:
                    patient_id = str(getattr(ds, 'PatientID', ''))
            except Exception:
                continue
        return kept, pixel_spacing, patient_id, sel_info

    # ── full DICOM metadata (match TANGERINE preprocessor tag set) ─────────
    def extract_dicom_meta(self, dcm_path):
        meta = {}
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        except Exception:
            return meta
        fields = ['PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
                  'PatientAge', 'StudyDate', 'StudyTime', 'SeriesDescription',
                  'SeriesInstanceUID', 'Modality', 'Manufacturer',
                  'ManufacturerModelName', 'SliceThickness', 'SpacingBetweenSlices',
                  'PixelSpacing', 'WindowCenter', 'WindowWidth',
                  'RescaleIntercept', 'RescaleSlope']
        for f in fields:
            try:
                v = getattr(ds, f, '')
                meta[f] = str(v) if v is not None else ''
            except Exception:
                meta[f] = ''
        return meta

    # ── load + window + resize a single slice ──────────────────────────────
    def load_slice(self, path):
        dcm = pydicom.dcmread(path)
        arr = apply_modality_lut(dcm.pixel_array, dcm)        # -> HU
        arr = apply_windowing(arr.astype(np.float64),
                              WINDOW_CENTER, WINDOW_WIDTH)      # -> [0, 65535]
        arr = arr // 256                                       # -> [0, 255] (8-bit parity)
        arr = arr.astype(np.float32)
        # resize to 256x256 (bilinear, matches albumentations A.Resize default)
        arr = cv2.resize(arr, (self.img_size[1], self.img_size[0]),
                         interpolation=cv2.INTER_LINEAR)
        return arr  # (H, W)

    # ── full CT ────────────────────────────────────────────────────────────
    def process_ct(self, dicom_folder, output_path):
        try:
            kept, pixel_spacing, patient_id, sel_info = self.filter_slices(dicom_folder)
            if len(kept) == 0:
                return {'success': False, 'error': 'no usable axial series (CT dropped)'}
            if len(kept) < 10:
                return {'success': False, 'error': f'too few slices ({len(kept)})'}
            if pixel_spacing is None:
                pixel_spacing = [1.0, 1.0]

            thicks = [th for (_, _, th) in kept if th is not None and th > 0]
            series_thick = sel_info.get('thickness')
            if series_thick is None and thicks:
                series_thick = float(np.median(thicks))

            # order slices by z
            kept.sort(key=lambda t: t[1])
            paths = [p for (p, _, _) in kept]

            # load all slices -> (T, H, W)
            slices = [self.load_slice(p) for p in paths]
            vol = np.stack(slices, axis=0).astype(np.float32)  # (T, H, W)

            # build (C=1, X=H, Y=W, Z=T) tensor for torchio, with original affine
            t = torch.from_numpy(vol).float()                  # (T, H, W)
            t = t.unsqueeze(0)                                  # (1, T, H, W) = (C,T,H,W)
            t = t.permute(0, 2, 3, 1)                           # (C, H, W, T) = (C,X,Y,Z)

            # original voxel spacing affine: pixel_spacing(x,y), thickness(z)=median dz
            zpos = [z for (_, z, _) in kept]
            dz = float(np.median(np.abs(np.diff(zpos)))) if len(zpos) > 1 else self.voxel_spacing[2]
            if dz <= 0:
                dz = self.voxel_spacing[2]
            affine = torch.diag(torch.tensor(
                [pixel_spacing[0], pixel_spacing[1], dz, 1.0], dtype=torch.float32))

            img = tio.ScalarImage(affine=affine.numpy(), tensor=t)
            img = self.resample_transform(img)                 # -> VOXEL_SPACING
            img = self.padding_transform(img)                  # -> (256,256,200)
            out = img.data[0].numpy()                           # (X=256, Y=256, Z=200)

            # save as nii.gz with sitk: array order (Z, Y, X)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_path = Path(output_path)
            if self.output_format == 'nifti':
                sitk_arr = np.transpose(out, (2, 1, 0)).astype(np.float32)  # (Z,Y,X)
                out_img = sitk.GetImageFromArray(sitk_arr)
                out_img.SetSpacing((self.voxel_spacing[0], self.voxel_spacing[1], self.voxel_spacing[2]))
                fd, tmp_name = tempfile.mkstemp(prefix=output_path.name + '.', suffix='.tmp.nii.gz', dir=str(output_path.parent))
                os.close(fd)
                os.unlink(tmp_name)
                try:
                    sitk.WriteImage(out_img, tmp_name)
                    _validate_nifti(tmp_name, expected_shape=sitk_arr.shape)
                    os.replace(tmp_name, output_path)
                    _validate_nifti(output_path, expected_shape=sitk_arr.shape)
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
            else:
                fd, tmp_name = tempfile.mkstemp(prefix=output_path.name + '.', suffix='.tmp.npz', dir=str(output_path.parent))
                os.close(fd)
                try:
                    np.savez_compressed(tmp_name, image=out.astype(np.float32))
                    if os.path.getsize(tmp_name) == 0:
                        raise IOError(f'Output file missing or empty: {tmp_name}')
                    os.replace(tmp_name, output_path)
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)

            result = {
                'success': True,
                'n_slices_kept': len(kept),
                'final_shape': list(out.shape),
                'orig_pixel_spacing': pixel_spacing,
                'orig_dz_median': dz,
                'voxel_spacing': list(self.voxel_spacing),
                'value_range': [float(out.min()), float(out.max())],
                'PatientID': patient_id,
                'series_selection': sel_info.get('selection'),
                'series_orientation': sel_info.get('orientation'),
                'series_n_total': sel_info.get('n_series_total'),
                'series_desc': sel_info.get('desc'),
            }
            # Full DICOM metadata (same tag set as TANGERINE preprocessor)
            result.update(self.extract_dicom_meta(paths[0]))
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ── batch (resumable) ────────────────────────────────────────────────────
    def scan_dicom_folders(self, data_root):
        data_path = Path(data_root)
        folders = []
        with os.scandir(data_path) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=True):
                    f = Path(entry.path)
                    if len(list(f.glob('*.dcm'))) > 0:
                        folders.append(f)
        print(f"[INFO] Found {len(folders)} DICOM folders")
        return sorted(folders, key=lambda p: p.name)

    def get_completed(self, output_root):
        output_root = Path(output_root)
        done = set()
        ext = 'nii.gz' if self.output_format == 'nifti' else 'npz'
        expected = (self.num_images, self.img_size[1], self.img_size[0]) if self.output_format == 'nifti' else (self.img_size[0], self.img_size[1], self.num_images)
        for log in [output_root / 'preprocessing_log.json', *sorted(output_root.glob('preprocessing_log_job_*.json'))]:
            if log.exists():
                try:
                    for r in json.load(open(log)):
                        name = r.get('folder_name', '')
                        if r.get('success') and name:
                            out_file = output_root / f'{name}.{ext}'
                            if _valid_existing_output(out_file, self.output_format, expected):
                                done.add(name)
                except Exception:
                    pass
        for out_file in output_root.glob(f'*.{ext}'):
            name = out_file.name[:-len(f'.{ext}')]
            if _valid_existing_output(out_file, self.output_format, expected):
                done.add(name)
        return done

    def batch(self, data_root, output_root, job_id=0, total_jobs=1, resume=True):
        out = Path(output_root); out.mkdir(parents=True, exist_ok=True)
        completed = self.get_completed(output_root) if resume else set()
        if completed:
            print(f"[INFO] {len(completed)} already done (resume)")

        folders = self.scan_dicom_folders(data_root)
        if not folders:
            print(f"[ERROR] No DICOM folders in {data_root}"); return []

        all_scans = [{'folder': f, 'folder_name': f.name} for f in folders]
        per = len(all_scans) // total_jobs
        s = job_id * per
        e = s + per if job_id < total_jobs - 1 else len(all_scans)
        job = all_scans[s:e]
        print(f"\n[JOB {job_id}/{total_jobs}] scans {s}..{e-1} (n={len(job)})")
        print(f"[INFO] Sybil-exact: <=2.5mm slices, window {WINDOW_CENTER}/{WINDOW_WIDTH}, "
              f"spacing {self.voxel_spacing}, {self.img_size}x{self.num_images}, 1-channel")
        print("=" * 60)

        results, skipped = [], 0
        ext = 'nii.gz' if self.output_format == 'nifti' else 'npz'
        for idx, sc in enumerate(job):
            if resume and sc['folder_name'] in completed:
                skipped += 1; continue
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"[PROGRESS] {idx+1}/{len(job)} ({skipped} skipped)"); sys.stdout.flush()
            of = out / f"{sc['folder_name']}.{ext}"
            r = self.process_ct(sc['folder'], of)
            r.update({'folder_name': sc['folder_name'], 'output_file': str(of), 'job_id': job_id})
            results.append(r)
            _atomic_json_dump(results, out / f'preprocessing_log_job_{job_id}.json')

        jl = out / f'preprocessing_log_job_{job_id}.json'
        _atomic_json_dump(results, jl)
        ok = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}\n[JOB {job_id}] done. processed={len(results)} success={ok} "
              f"failed={len(results)-ok} skipped={skipped}\n{'='*60}")
        return results


def merge_job_logs(output_root, total_jobs):
    import pandas as pd
    out = Path(output_root); rows = []
    for j in range(total_jobs):
        jl = out / f'preprocessing_log_job_{j}.json'
        if jl.exists():
            rows.extend(json.load(open(jl)))
    if not rows:
        print("[WARNING] no logs"); return
    _atomic_json_dump(rows, out / 'preprocessing_log.json')
    pd.DataFrame(rows).to_csv(out / 'preprocessing_log.csv', index=False)
    ok = sum(1 for r in rows if r['success'])
    print(f"[MERGE] {len(rows)} scans, {ok} success")


def main():
    p = argparse.ArgumentParser(description='Sybil-exact 3D CT preprocessing (1 file/CT)')
    p.add_argument('--data_root')
    p.add_argument('--output_root', required=True)
    p.add_argument('--job_id', type=int, default=0)
    p.add_argument('--total_jobs', type=int, default=1)
    p.add_argument('--max_slice_thickness', type=float, default=MAX_SLICE_THICKNESS)
    p.add_argument('--format', choices=['nifti', 'npz'], default='nifti')
    p.add_argument('--no_resume', action='store_true')
    p.add_argument('--merge_logs', action='store_true')
    args = p.parse_args()

    if not args.merge_logs and not args.data_root:
        p.error('--data_root is required unless --merge_logs is used')

    if args.merge_logs:
        merge_job_logs(args.output_root, args.total_jobs); return

    pre = SybilExact3DPreprocessor(
        max_slice_thickness=args.max_slice_thickness,
        output_format=args.format)
    pre.batch(args.data_root, args.output_root, args.job_id, args.total_jobs, not args.no_resume)


if __name__ == '__main__':
    main()
