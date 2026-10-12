"""10-fold patient-level CV assignment, stratified by cancer x event-year bin.
Reproducible: fixed seed (42) + assignment persisted to folds.csv.
Balances outcome + cancer-timing by stratification; age/sex/scanner verified."""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
CV = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260615/cross_val"
SRC = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260615/split2/dataset_splits"
META = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_tan_syb/ct_metadata/ct_metadata_all.csv"
SEED = 42
os.makedirs(CV, exist_ok=True)

full = pd.concat([pd.read_csv(f"{SRC}/{s}.csv") for s in ["train", "val", "test"]], ignore_index=True)
full["pat_id"] = full["pat_id"].astype(str)
print(f"full cohort: {len(full)} scans, {full['pat_id'].nunique()} patients, "
      f"{int((full.groupby('pat_id')['cancer'].max()==1).sum())} cancer patients\n")

pat = full.groupby("pat_id").agg(cancer=("cancer", "max"), time=("time_at_event", "first")).reset_index()
pat["strat"] = pat.apply(lambda r: f"pos_y{int(r.time)}" if r.cancer == 1 else "neg", axis=1)
counts = pat["strat"].value_counts()
pat["strat"] = pat["strat"].apply(lambda s: s if (s == "neg" or counts[s] >= 10) else "pos_other")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
pat["fold"] = -1
for fold_id, (_, test_idx) in enumerate(skf.split(pat, pat["strat"])):
    pat.iloc[test_idx, pat.columns.get_loc("fold")] = fold_id
pat["fold"] = pat["fold"].astype(int)
pat[["pat_id", "fold", "strat", "cancer"]].to_csv(f"{CV}/folds.csv", index=False)

full["fold"] = full["pat_id"].map(dict(zip(pat.pat_id, pat.fold)))
m = pd.read_csv(META, dtype=str, usecols=["folder_name", "Manufacturer"])
sie = dict(zip(m["folder_name"], m["Manufacturer"].str.upper().str.contains("SIEMENS")))
full["siemens"] = full["ct_id"].astype(str).map(sie).fillna(False)

print("=" * 78); print("PER-FOLD BALANCE"); print("=" * 78)
rows = []
for f in range(10):
    g = full[full.fold == f]; pc = g.groupby("pat_id")["cancer"].max()
    rows.append(dict(fold=f, patients=g["pat_id"].nunique(), scans=len(g),
                     cancer_pat=int((pc == 1).sum()), pos_rate=round(100 * (pc == 1).mean(), 2),
                     age=round(pd.to_numeric(g.groupby("pat_id")["age"].first(), errors="coerce").mean(), 1),
                     pct_M=round(100 * (g.groupby("pat_id")["sex"].first().astype(str).str[0].str.upper() == "M").mean(), 1),
                     siemens_pct=round(100 * g["siemens"].mean(), 1)))
bal = pd.DataFrame(rows); print(bal.to_string(index=False))
print("\nspread across folds (max-min): pos_rate=%.2f  age=%.1f  pctM=%.1f  siemens=%.1f" % (
    bal.pos_rate.max() - bal.pos_rate.min(), bal.age.max() - bal.age.min(),
    bal.pct_M.max() - bal.pct_M.min(), bal.siemens_pct.max() - bal.siemens_pct.min()))

print("\n" + "=" * 78); print("PER-FOLD x PER-YEAR positive scans (cumulative)"); print("=" * 78)
print("fold  " + " ".join(f"Y{k+1:>4}" for k in range(6)))
for f in range(10):
    g = full[full.fold == f]; c = g["cancer"].values; t = g["time_at_event"].values
    yr = [int(((c == 1) & (t <= k)).sum()) for k in range(6)]
    print(f"{f:>4}  " + " ".join(f"{v:>5}" for v in yr))
print(f"\nReproducible: seed={SEED}, assignment saved to {CV}/folds.csv")
