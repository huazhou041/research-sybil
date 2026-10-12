"""Sybil-1.5 predictions -> all_predictions CSV in the sybil-comparable schema.

Input : cv_eval/preds/seed{S}/checkpoints/3/test.csv
        columns: accession, time_at_event, y, survival="[l1 ... l6]"  (LOGITS)

Usage :
  # one model (do this for each finished seed)
  python merge_predictions.py --single 0
  # ensemble of whatever seeds are done (later)
  python merge_predictions.py --ensemble 0,1,2

Output: cv_eval/all_predictions_seed{S}.csv  /  all_predictions_ensemble.csv
        split, patient_id, ct_id, cancer, time_at_event, fold,
        pred_1..pred_6 (calibrated), pred_1_raw..pred_6_raw (sigmoid of logits)

LEAKAGE CONTROL
  * Model frozen (NLST-trained, external); no cohort scan trains or tunes it.
  * pred_k_raw = sigmoid(logit_k) -- pure model output, nothing fit on this cohort.
  * pred_k     = CROSS-FITTED isotonic: fold f's year-k calibrator is fit ONLY on scans
                 outside fold f, then applied to fold f. A scan's calibrated score never
                 sees its own label nor any label from its own test fold.
  * AUC is rank-based and calibration is monotonic => per-year AUC is identical for raw
    vs calibrated. Calibration only makes the probabilities interpretable.
"""
import argparse, os, re
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

CV    = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/v15_inference/cv_eval"
BASE  = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260615"
FOLDS = f"{BASE}/cross_val/folds"
NF, MAXT = 10, 6


def _logits(s):
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", str(s))]
    if len(v) != MAXT:
        raise ValueError(f"expected {MAXT} logits, got {len(v)}: {s!r}")
    return v


def load_seed(seed):
    path = f"{CV}/preds/seed{seed}/checkpoints/3/test.csv"
    if not os.path.exists(path):
        raise SystemExit(f"missing predictions for seed{seed}: {path}")
    df = pd.read_csv(path)
    L = np.array([_logits(s) for s in df["survival"]], dtype=float)
    out = pd.DataFrame({
        "ct_id": df["accession"].astype(str),
        "cancer": df["y"].astype(int),
        "time_at_event": df["time_at_event"].astype(float),
    })
    # sigmoid -> per-year probability
    for k in range(MAXT):
        out[f"p{k+1}"] = 1.0 / (1.0 + np.exp(-L[:, k]))
    n0 = len(out)
    out = out.drop_duplicates("ct_id").sort_values("ct_id").reset_index(drop=True)
    print(f"seed{seed}: {n0} rows -> {len(out)} unique scans")
    return out


def fold_map():
    m = {}
    for f in range(NF):
        d = pd.read_csv(f"{FOLDS}/fold{f}/test.csv", usecols=["ct_id"])
        for c in d["ct_id"].astype(str):
            m[c] = f
    return m


def cross_fitted_calibration(df):
    """Per-year isotonic fit OUT-OF-FOLD (leakage-free). Returns (n, 6)."""
    cancer = df["cancer"].values.astype(int)
    time   = df["time_at_event"].values.astype(float)
    fold   = df["fold"].values
    cal = np.empty((len(df), MAXT))
    for k in range(MAXT):
        raw = df[f"pred_{k+1}_raw"].values
        pos = (cancer == 1) & (time <= k)
        neg = (cancer == 0) & (time >= k)
        valid, lab = pos | neg, pos.astype(int)
        for f in range(NF):
            te = fold == f
            tr = valid & (fold != f)
            if tr.sum() < 20 or lab[tr].sum() in (0, int(tr.sum())):
                cal[te, k] = raw[te]           # not enough signal -> leave raw
            else:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(raw[tr], lab[tr])
                cal[te, k] = iso.predict(raw[te])
    return cal


def build(seeds, tag):
    frames = [load_seed(s) for s in seeds]
    ids = frames[0]["ct_id"].values
    for s, d in zip(seeds[1:], frames[1:]):
        if not np.array_equal(ids, d["ct_id"].values):
            raise SystemExit(f"seed{s} ct_id set differs from seed{seeds[0]}")
    # ensemble = mean of per-seed PROBABILITIES
    P = np.mean([d[[f"p{k+1}" for k in range(MAXT)]].values for d in frames], axis=0)

    out = frames[0][["ct_id", "cancer", "time_at_event"]].copy()
    for k in range(MAXT):
        out[f"pred_{k+1}_raw"] = P[:, k]

    out["fold"] = out["ct_id"].map(fold_map())
    n_un = int(out["fold"].isna().sum())
    out = out[out["fold"].notna()].copy()
    out["fold"] = out["fold"].astype(int)
    print(f"mapped to folds: {len(out)}   (dropped not-in-cohort: {n_un})")

    cal = cross_fitted_calibration(out)
    for k in range(MAXT):
        out[f"pred_{k+1}"] = cal[:, k]

    out["patient_id"], out["split"] = out["ct_id"], "test"
    cols = (["split", "patient_id", "ct_id", "cancer", "time_at_event", "fold"]
            + [c for k in range(MAXT) for c in (f"pred_{k+1}", f"pred_{k+1}_raw")])
    path = f"{CV}/all_predictions_{tag}.csv"
    out[cols].to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}  cancer+={int(out.cancer.sum())}")

    # Flatness diagnostic: old Sybil saturated its CumulativeProbabilityLayer so every
    # year got an identical score. Check whether Sybil-1.5 does the same here.
    R = out[[f"pred_{k+1}_raw" for k in range(MAXT)]].values
    allsame = np.mean(np.all(np.isclose(R, R[:, [0]], atol=1e-6), axis=1))
    print(f"[flatness] scans with ALL 6 years identical : {allsame:.2%}")
    print(f"[flatness] mean per-scan spread (max-min)   : {np.mean(R.max(1)-R.min(1)):.6f}")
    print(f"[flatness] per-year mean prob              : "
          + " ".join(f"Y{k+1}={R[:,k].mean():.4f}" for k in range(MAXT)))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--single", type=int, help="one seed, e.g. --single 0")
    g.add_argument("--ensemble", type=str, help="comma seeds, e.g. --ensemble 0,1,2")
    a = ap.parse_args()
    if a.single is not None:
        build([a.single], f"seed{a.single}")
    else:
        seeds = [int(x) for x in a.ensemble.split(",")]
        build(seeds, "ensemble")
