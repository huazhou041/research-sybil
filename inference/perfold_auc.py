"""Per-fold x per-year AUC for Sybil-1.5, following cross_v3_fusion's exact logic.

AUC uses CUMULATIVE masking (identical to cross_v3_fusion / perfold_peryear_auc.py):
    for year k:  pos = cancer==1 & time_at_event <= k
                 neg = cancer==0 & time_at_event >= k
                 AUC = roc_auc_score(pos, prob_k)  over pos|neg only

Sybil-1.5 is FROZEN, so scores come from ONE prediction pass and the 10 CV folds only
PARTITION the 14,549 scans for evaluation -- no per-fold training, no leakage.

Reports, for BOTH calibrated (pred_k) and raw (pred_k_raw):
    f0..f9 x Y1..Y6, each fold's avg-over-years,
    AVG    = mean of the 10 folds   (the 10-fold average)
    POOLED = all scans scored at once

Usage:
  python perfold_auc.py --pred all_predictions_seed0.csv --model "Sybil-1.5 seed0" \
                        --out_csv perfold_seed0.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CV    = Path("/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/v15_inference/cv_eval")
BASE  = Path("/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260615")
FOLDS = BASE / "cross_val/folds"
NF, MAXT = 10, 6


def auc_year(cancer, time, prob_k, k):
    pos = (cancer == 1) & (time <= k)
    neg = (cancer == 0) & (time >= k)
    m = pos | neg
    yt, yp = pos[m].astype(int), prob_k[m]
    if yt.sum() == 0 or yt.sum() == len(yt):
        return np.nan
    return roc_auc_score(yt, yp)


def load_fold_map():
    fmap = {}
    for f in range(NF):
        d = pd.read_csv(FOLDS / f"fold{f}" / "test.csv", usecols=["ct_id"])
        for c in d["ct_id"].astype(str):
            fmap[c] = f
    return fmap


def compute(df, tpl):
    c = df["cancer"].values.astype(int)
    t = df["time_at_event"].values.astype(float)
    fold = df["fold"].values
    M = np.full((NF, MAXT), np.nan)
    for f in range(NF):
        idx = fold == f
        for k in range(MAXT):
            M[f, k] = auc_year(c[idx], t[idx], df[tpl.format(k + 1)].values[idx], k)
    pooled = np.array([auc_year(c, t, df[tpl.format(k + 1)].values, k) for k in range(MAXT)])
    return M, np.nanmean(M, axis=0), pooled


def show(name, M, avg, pooled):
    print(f"\n===== {name}")
    print("fold  " + " ".join(f"Y{k+1:>6}" for k in range(MAXT)) + "   avgYrs")
    for f in range(NF):
        row = " ".join(f"{M[f,k]:6.3f}" if not np.isnan(M[f, k]) else "   n/a" for k in range(MAXT))
        print(f"f{f:<3} {row}   {np.nanmean(M[f]):6.3f}")
    print("AVG  " + " ".join(f"{avg[k]:6.3f}" for k in range(MAXT)) + f"   {np.nanmean(avg):6.3f}")
    print("POOL " + " ".join(f"{pooled[k]:6.3f}" for k in range(MAXT)) + f"   {np.nanmean(pooled):6.3f}")


def rows_for_csv(model, kind, M, avg, pooled):
    out = [[model, kind, f"f{f}"] + [M[f, k] for k in range(MAXT)] + [np.nanmean(M[f])]
           for f in range(NF)]
    out.append([model, kind, "AVG"] + list(avg) + [np.nanmean(avg)])
    out.append([model, kind, "POOLED"] + list(pooled) + [np.nanmean(pooled)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="all_predictions_*.csv (name or full path)")
    ap.add_argument("--model", default="Sybil-1.5")
    ap.add_argument("--out_csv", default="")
    a = ap.parse_args()

    p = Path(a.pred)
    if not p.exists():
        p = CV / a.pred
    df = pd.read_csv(p)
    df["ct_id"] = df["ct_id"].astype(str) if "ct_id" in df.columns else df["patient_id"].astype(str)
    if "fold" not in df.columns:
        df["fold"] = df["ct_id"].map(load_fold_map())
    n_un = int(df["fold"].isna().sum())
    df = df[df["fold"].notna()].copy()
    df["fold"] = df["fold"].astype(int)
    print(f"{a.model}: {len(df)} scans mapped to folds (unmapped: {n_un}), "
          f"cancer+={int(df.cancer.sum())}")

    csv_rows = []
    for kind, tpl in [("calibrated", "pred_{}"), ("raw", "pred_{}_raw")]:
        if tpl.format(1) not in df.columns:
            continue
        M, avg, pooled = compute(df, tpl)
        show(f"{a.model} [{kind}]", M, avg, pooled)
        csv_rows += rows_for_csv(a.model, kind, M, avg, pooled)

    if a.out_csv:
        out = Path(a.out_csv)
        if not out.is_absolute():
            out = CV / out
        cols = ["model", "kind", "fold"] + [f"Y{k+1}" for k in range(MAXT)] + ["avgYrs"]
        pd.DataFrame(csv_rows, columns=cols).to_csv(out, index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
