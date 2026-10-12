"""Build res_20260801/report_v3.html for the axial-fix rerun.

Reuses the EXISTING (pre-axial-fix) report_v3.html as a shell -- same technique
build_report.py already uses one generation up (reusing sybil_v1_reprocess's
report.html as a shell for cohort covariates). report_v3.html carries the
inlined Plotly bundle, the cohort covariates (lrads, manufacturer, sex, smoke,
age, dates, ctdivol, chest-protocol family -- all joined from cluster-side
metadata not held locally), AND the ROI-modal JS/CSS framework
(openRoiModal/closeRoiModal, .roi-backdrop/.roi-modal styles, the plotly_click
handler). All of that is independent of which preprocessing produced the pixel
data -- same 14,549 scans, same order-independent ct_id join -- so it transfers
unchanged.

What this script replaces:
  - DATA.methods -- rebuilt from the new axial-fix full-precision prediction
    CSVs in ../predictions/ (same 6 models, same schema as build_report.py).
  - ROI_MAP -- the set of cancer ct_ids with a causally-validated ROI PNG.
    If ../roi_images/ is not yet populated (v15 ROI occlusion + render jobs
    still running on cluster), this is written as {} so the report is fully
    valid HTML with the modal simply inert (no clickable cells) until rerun.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULT = HERE.parent  # res_20260801 itself
PRED = RESULT / "predictions"
OLD_RESULT = HERE.parent.parent / "result"
SHELL = OLD_RESULT / "report_v3.html"
ROI_DIR = RESULT / "roi_images"
DST = RESULT / "report_v3.html"
MAXT = 6

MASK_NOTE = (
    "Masking (positives are the same either way: cancer with time≤k). "
    "PURE = negatives are cancer-free with follow-up≥k, so anyone diagnosed after "
    "year k is dropped from the comparison. CUMULATIVE = negatives are everyone "
    "still at risk at year k who is not yet a positive, so those later-diagnosed "
    "patients are counted as controls — which penalises a model for scoring them "
    "high and usually gives the lower AUC. Pooled AUC quoted here uses PURE masking."
)

MODELS = [
    ("sybil_v1_reprocess single", "sybil_v1_reprocess_single.csv", "patient_id",
     "Single pretrained Sybil checkpoint, predict-only (no training), on v1-reprocess "
     "reference-faithful volumes (z_spacing=SliceThickness, pad=128.1722, corrected "
     "in-plane axis order), AXIAL-FIX cohort (coronal-MPR-misclassified-as-axial "
     "series-selection bug fixed). Split-independent."),
    ("sybil_v1_reprocess ensemble", "sybil_v1_reprocess_ensemble.csv", "patient_id",
     "Official 5-checkpoint Sybil ENSEMBLE, predict-only, on the same AXIAL-FIX "
     "v1-reprocess volumes; raw probabilities averaged across checkpoints then "
     "calibrated. Split-independent."),
    ("sybil_v15 seed0", "sybil_v15_seed0.csv", "ct_id",
     "Released Sybil-1.5 checkpoint seed0 (YalaLab/pillar-0 on HuggingFace; finetuned "
     "from Pillar-0 by its authors on NLST — NOT trained on any NYU data). Run here "
     "predict-only (--evaluate, every row split=test) on the AXIAL-FIX rave-native "
     "384x384 RVE volumes: one frozen inference pass, folds partition for evaluation only."),
    ("sybil_v15 seed1", "sybil_v15_seed1.csv", "ct_id",
     "Released Sybil-1.5 checkpoint seed1, same external provenance and protocol as seed0."),
    ("sybil_v15 seed2", "sybil_v15_seed2.csv", "ct_id",
     "Released Sybil-1.5 checkpoint seed2, same external provenance and protocol as seed0."),
    ("sybil_v15 ensemble", "sybil_v15_ensemble.csv", "ct_id",
     "Mean of the three Sybil-1.5 seeds' probabilities, then cross-fitted isotonic "
     "calibration. NOTE: single seeds are heavily tied (bf16-mixed autocast at inference, "
     "~8-13% distinct values); the ensemble is the only run with per-scan resolution."),
]


def load_flat(path, key, ct_order):
    s = pd.read_csv(path)
    ids = s[key].astype(str).tolist()
    calm = np.round(s[[f"pred_{t + 1}" for t in range(MAXT)]].values.astype(float), 4)
    rawm = np.round(s[[f"pred_{t + 1}_raw" for t in range(MAXT)]].values.astype(float), 4)
    pos = {c: i for i, c in enumerate(ids)}
    idx = np.array([pos.get(c, -1) for c in ct_order])
    miss = int((idx < 0).sum())
    safe = np.where(idx < 0, 0, idx)
    cal, raw = calm[safe], rawm[safe]
    fc = cal.tolist()
    fr = raw.tolist()
    if miss:
        for r in np.flatnonzero(idx < 0):
            fc[r] = [None] * MAXT
            fr[r] = [None] * MAXT
    return [v for row in fc for v in row], [v for row in fr for v in row], miss


def pooled_auc(scores, cancer, time, n):
    s = np.asarray([np.nan if x is None else x for x in scores], dtype=float)
    s = s.reshape(n, MAXT)
    c, t = np.asarray(cancer), np.asarray(time)
    vals = []
    for k in range(MAXT):
        pos = (c == 1) & (t <= k)
        neg = (c == 0) & (t >= k)
        sel = pos | neg
        y, sc = pos[sel].astype(int), s[sel, k]
        ok = ~np.isnan(sc)
        y, sc = y[ok], sc[ok]
        npos, nneg = int((y == 1).sum()), int((y == 0).sum())
        if npos == 0 or nneg == 0:
            continue
        r = pd.Series(sc).rank().values
        vals.append((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return round(float(np.mean(vals)), 3) if vals else None


def main():
    shell = SHELL.read_text()
    i = shell.index("const DATA = ")
    j = shell.index("\nconst ROI_MAP")
    data = json.loads(shell[i + len("const DATA = "):j].rstrip().rstrip(";"))
    ct_order = [str(c) for c in data["ct_id"]]
    print(f"shell cohort: n={data['n']}  covariates+ctdivol+chestfam carried over unchanged")

    methods = []
    for name, fn, key, desc in MODELS:
        fc, fr, miss = load_flat(PRED / fn, key, ct_order)
        pooled = pooled_auc(fc, data["cancer"], data["time"], data["n"])
        full_desc = desc if pooled is None else f"{desc} (pooled AUC {pooled:.3f})"
        full_desc = f"{full_desc}  |  {MASK_NOTE}"
        methods.append({"name": name, "pooled": pooled, "desc": full_desc,
                        "scores": fc, "scores_raw": fr, "missing": miss})
        print(f"  {name:30s} missing_in_cohort={miss}  pooled={pooled}")
    data["methods"] = methods

    # ---- ROI_MAP: cancer ct_ids with a rendered ROI PNG in ../roi_images/ ----
    cancer_ids = {c for c, ct in zip(data["ct_id"], data["cancer"]) if ct == 1}
    if ROI_DIR.is_dir():
        have = {p.stem for p in ROI_DIR.glob("*.png")}
        roi_ids = sorted(str(c) for c in cancer_ids if str(c) in have)
    else:
        roi_ids = []
    roi_map_js = "{" + ",".join(f'"{c}":1' for c in roi_ids) + "}"
    print(f"ROI_MAP: {len(roi_ids)} cancer scans with a rendered ROI image"
          f"{' (roi_images/ not yet populated -- run again once the cluster render job finishes)' if not roi_ids else ''}")

    k0 = shell.index("const ROI_MAP = ")
    k1 = shell.index("\n", k0)
    new_roi_line = f"const ROI_MAP = {roi_map_js};  // ct_id -> 1; {len(roi_ids)} cancer scans w/ causally-validated ROI image"

    head, tail = shell[:i], shell[j:]
    out = head + "const DATA = " + json.dumps(data, separators=(",", ":")) + tail
    # re-locate ROI_MAP line in the freshly-assembled `out` (offsets shifted vs `shell`)
    k0 = out.index("const ROI_MAP = ")
    k1 = out.index("\n", k0)
    out = out[:k0] + new_roi_line + out[k1:]

    subs = [
        ("<title>sybil_v15 vs sybil_v1_reprocess - 6-year lung-cancer risk</title>",
         "<title>sybil_v15 vs sybil_v1_reprocess - AXIAL-FIX rerun (2026-08-01)</title>"),
        ("<h1>Sybil 6-year Lung-Cancer Risk &mdash; sybil_v15 vs sybil_v1_reprocess</h1>",
         "<h1>Sybil 6-year Lung-Cancer Risk &mdash; sybil_v15 vs sybil_v1_reprocess "
         "(AXIAL-FIX rerun)</h1>"),
    ]
    for a, b in subs:
        if a not in out:
            raise SystemExit(f"shell text not found, cannot patch safely:\n  {a[:90]}")
        out = out.replace(a, b)

    # footer: note this is the axial-fix rerun, not the original cohort
    k = out.index("<footer>")
    k2 = out.index("</footer>", k) + len("</footer>")
    out = out[:k] + (
        "<footer>Generated from the AXIAL-FIX rerun (coronal-MPR-misclassified-as-axial "
        "DICOM series-selection bug fixed, models/sybil_v15_20260731) &mdash; self-contained "
        "(Plotly inlined). Data: 14,549 scans, 6 models. Cohort covariates (lrads, "
        "manufacturer, sex, smoke, age, dates, CTDIvol, chest-protocol family) carried "
        "over unchanged from the pre-fix report since they describe the same patients/scans "
        "and are independent of series selection. ROI images (patch-embed Grad-CAM + causal "
        "occlusion, roi_occlusion_v15, seeds 0 &amp; 2) reflect the axial-fix data; "
        "heatmap cells for cancer scans with a validated ROI are clickable.</footer>"
    ) + out[k2:]

    DST.write_text(out)
    print(f"-> {DST}  ({DST.stat().st_size / 1e6:.1f} MB)  models={len(methods)}  roi_map={len(roi_ids)}")


if __name__ == "__main__":
    main()
