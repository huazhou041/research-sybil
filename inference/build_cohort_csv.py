"""Build the leakage-free Sybil-1.5 inference CSV over the 14,549 CV cohort.

Sybil-1.5 (Pillar0) is trained on NLST (external) and frozen here, so inference
is split-independent: we mark every scan split=test, predict once, and let the
10 CV folds (cross_val/folds) partition scans for evaluation -- identical design
to the frozen sybil-v2 reference (predict_sybil.py / perfold_peryear_auc.py).
No cohort scan ever trains the model => no leakage.

AXIAL-FIX: RVE now points at images_3d_sybil_axial/sybil_v15 (coronal-MPR-
misclassified-as-axial series-selection bug fixed), not the original
images_3d_syb_v2/syb_v15. Same split/label source (dataset_splits, copied
into v1_inference/ for this bundle -- it only carries labels, independent of
which preprocessing produced the pixel data).
"""
import os, glob
import pandas as pd

B   = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/v1_inference"
RVE = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_sybil_axial/sybil_v15"
OUT = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/v15_inference/cv_eval/cohort_test.csv"

# cohort = union of the seed-42 patient split (train+val+test) = the 14,549 CV cohort
rows = []
for s in ["train", "val", "test"]:
    df = pd.read_csv(f"{B}/dataset_splits/{s}.csv",
                     usecols=["ct_id", "cancer", "time_at_event", "pat_id"])
    rows.append(df)
coh = pd.concat(rows, ignore_index=True)
coh["ct_id"] = coh["ct_id"].astype(str)
n0 = len(coh)
coh = coh.drop_duplicates("ct_id")
print(f"cohort rows {n0} -> unique ct_id {len(coh)}")

# available RVE folders
have = {os.path.basename(d)[:-4] for d in glob.glob(RVE + "/*.1.0")}
coh["image_paths"] = coh["ct_id"].map(lambda c: f"{RVE}/{c}.1.0")
present = coh["ct_id"].isin(have)
missing = coh[~present]
print(f"present in syb_v15 RVE: {int(present.sum())}   missing: {len(missing)}")
if len(missing):
    print("  missing ct_ids:", list(missing['ct_id'])[:20])

out = coh[present].copy()
out = out.rename(columns={"cancer": "y"})
out["accession"] = out["ct_id"]
out["split"] = "test"
out = out[["accession", "image_paths", "split", "y", "time_at_event", "ct_id", "pat_id"]]
# y and time_at_event as ints
out["y"] = out["y"].astype(int)
out["time_at_event"] = out["time_at_event"].astype(int)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_csv(OUT, index=False)
print(f"wrote {OUT}  ({len(out)} rows)")
print("cancer+ :", int(out['y'].sum()), " patients:", out['pat_id'].nunique())
print(out.head(3).to_string())
