"""
Shared DICOM series selection for TANGERINE + Sybil preprocessing.

Raw CT folders frequently contain MANY series mixed together:
  - axial thin-slice diagnostic (0.625-1.25mm)   <- what we want
  - axial thick (2.5-5mm)
  - sagittal / coronal reformats                  <- wrong orientation
  - scout / localizer (2-3 huge-thickness images) <- junk

A naive GetGDCMSeriesFileNames(dir) returns only the FIRST series (arbitrary),
and a naive per-slice thickness filter MIXES orientations. Both corrupt the
volume. select_best_series() picks ONE coherent series so both preprocessors
operate on the SAME diagnostic acquisition.

Policy (in order):
  1. Enumerate series via SimpleITK GetGDCMSeriesIDs
  2. Classify orientation from ImageOrientationPatient (axial only)
  3. Drop tiny series (< min_slices -> scouts/localizers)
  4. Among AXIAL series, prefer slice thickness <= prefer_thick_max (2.5mm),
     then choose the one with the MOST slices (most complete volume)
  5. Fallbacks: largest axial -> largest series overall
"""
import numpy as np
import pydicom
import SimpleITK as sitk


def _orientation(iop):
    """Classify slice orientation from ImageOrientationPatient (6 floats).
    Returns 'axial' | 'sagittal' | 'coronal' | 'unknown'."""
    try:
        row = np.array(iop[:3], dtype=float)
        col = np.array(iop[3:6], dtype=float)
        normal = np.cross(row, col)
        ax = np.argmax(np.abs(normal))   # 0=x(sag) 1=y(cor) 2=z(axial)
        return {0: 'sagittal', 1: 'coronal', 2: 'axial'}[int(ax)]
    except Exception:
        return 'unknown'


def select_best_series(dicom_folder, min_slices=10, prefer_thick_max=2.5,
                       verbose=False):
    """
    Return (file_list, info) for the best diagnostic series in a folder.

    Args:
        dicom_folder: path to a folder of .dcm files (possibly many series)
        min_slices: drop series with fewer files (scouts/localizers)
        prefer_thick_max: prefer axial series with thickness <= this (mm)

    Returns:
        (files, info_dict). Raises ValueError if no usable series.
    """
    r = sitk.ImageSeriesReader()
    series_ids = r.GetGDCMSeriesIDs(str(dicom_folder))
    if len(series_ids) == 0:
        # No multi-series info — fall back to whatever the reader finds
        files = r.GetGDCMSeriesFileNames(str(dicom_folder))
        if len(files) == 0:
            raise ValueError(f"No DICOM series in {dicom_folder}")
        return list(files), {'series_uid': None, 'orientation': 'unknown',
                             'thickness': None, 'n_files': len(files),
                             'selection': 'single_or_unknown'}

    candidates = []
    for sid in series_ids:
        files = r.GetGDCMSeriesFileNames(str(dicom_folder), sid)
        if len(files) < min_slices:
            continue  # scout / localizer / stub
        try:
            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            iop = getattr(ds, 'ImageOrientationPatient', None)
            orient = _orientation(iop) if iop is not None else 'unknown'
            thick = getattr(ds, 'SliceThickness', None)
            thick = float(thick) if thick is not None else None
            desc = str(getattr(ds, 'SeriesDescription', ''))
        except Exception:
            orient, thick, desc = 'unknown', None, ''
        candidates.append({'series_uid': sid, 'files': files, 'n_files': len(files),
                           'orientation': orient, 'thickness': thick, 'desc': desc})

    if not candidates:
        # everything was tiny — take the largest raw series as last resort
        best_sid = max(series_ids, key=lambda s: len(r.GetGDCMSeriesFileNames(str(dicom_folder), s)))
        files = r.GetGDCMSeriesFileNames(str(dicom_folder), best_sid)
        return list(files), {'series_uid': best_sid, 'orientation': 'unknown',
                             'thickness': None, 'n_files': len(files),
                             'selection': 'largest_no_candidates'}

    axial = [c for c in candidates if c['orientation'] == 'axial']
    pool = axial if axial else candidates
    selection = 'axial' if axial else 'non_axial_fallback'

    # Prefer thin (<= prefer_thick_max); among those, most slices
    thin = [c for c in pool if c['thickness'] is not None and 0 < c['thickness'] <= prefer_thick_max]
    if thin:
        best = max(thin, key=lambda c: c['n_files'])
        selection += '_thin_largest'
    else:
        best = max(pool, key=lambda c: c['n_files'])
        selection += '_largest'

    info = {'series_uid': best['series_uid'], 'orientation': best['orientation'],
            'thickness': best['thickness'], 'n_files': best['n_files'],
            'desc': best['desc'], 'n_series_total': len(series_ids),
            'selection': selection}
    if verbose:
        print(f"  [series-select] {len(series_ids)} series -> chose "
              f"{best['n_files']} files, {best['orientation']}, "
              f"thick={best['thickness']}, [{best['desc']}] ({selection})")
    return list(best['files']), info
