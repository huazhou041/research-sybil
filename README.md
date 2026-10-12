# research-sybil

Code for a retrospective comparison of Sybil v1 and Sybil-1.5 (built on the
Pillar-0/Atlas architecture) for lung-cancer-risk prediction on an NYU LDCT
screening cohort. **This repository contains pipeline code only — no patient
data, images, model weights, or predictions.** All paths referenced inside
the scripts point to internal cluster/share locations that are not included
here.

## Layout

- **`preprocessing/`** — raw DICOM → model-ready volume pipeline.
  `series_selection.py` picks the best series per study (axial-only,
  ≥10 slices, prefers ≤2.5mm thickness); `preprocess_sybil3d.py` /
  `preprocess_tangerine3d.py` apply the per-slice thickness filter and
  resample to the target voxel spacing; `batch_preprocess_*.sh` are the
  actual job invocations used to build the cohort; the `validate_*`/
  `extract_ct_metadata.py`/`check_no_usable_sybil_metadata.py` scripts are
  QC utilities over the resulting header metadata.

- **`cohort_split/`** — cohort assembly and fold construction.
  `prepare_survival_dataset.py` and `prepare_split.sh` build the modeling
  cohort (intersection of successfully-preprocessed TANGERINE and Sybil
  image sets, patient-level). `make_folds.py` does the 10-fold
  `StratifiedKFold` split (stratified by cancer × event-year bin).

- **`inference/`** — running the trained models on the cohort.
  `predict_sybil*.py`/`.sh` are the Sybil v1 inference entry points
  (single model + ensemble). `build_cohort_csv.py`/`merge_predictions.py`/
  `perfold_auc.py` are the Sybil-1.5 (v15) per-fold prediction and
  per-fold isotonic-recalibration/evaluation utilities.

- **`roi_attention/`** — Grad-CAM / occlusion / native-attention ROI panel
  generation for both model families. `roi_occlusion_v1.py` and
  `roi_native_attention_v1.py` run causal-occlusion-validated Grad-CAM and
  native-attention extraction for Sybil v1; `roi_occlusion_v15.py` does the
  Sybil-1.5 equivalent (per-seed). `render_roi_panel.py` composes the raw
  per-scan output into the box/gradient comparison panels used in the
  report. The `.sbatch` files are the actual SLURM job submissions.

- **`report_builders/`** — generates the self-contained HTML report from
  the cohort/prediction CSVs. One script per report version
  (`build_report_v3.py` → `build_report_v7.py`, each building on the
  previous version's output rather than the raw data); `build_sweep_report.py`/
  `sweep_compute.py` build a separate hyperparameter-sweep report.

- **`alicia_cohort/`** — joins an external patient roster (LCSP) against
  the modeling cohort by accession number to compute cohort overlap.
  `build_alicia_intersection.py` is the current (Python) version;
  `process.R` is an earlier, partial version kept for reference.

## What's intentionally not here

Model weights/checkpoints, DICOM/NIfTI images, ROI panel images, any CSV
containing patient-level data (IDs, dates, predictions, demographics), and
third-party vendor model code (e.g. the Pillar-0/Atlas HuggingFace model
definition files) are all excluded. Sybil v1 builds on the original
[Sybil package](https://github.com/reginabarzilaygroup/Sybil); Sybil-1.5
builds on the Pillar-0/Atlas architecture — neither is vendored here.
