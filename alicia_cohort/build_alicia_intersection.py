"""Complete, correct replacement for process.R.

process.R (still present in this folder) only went as far as flagging the
roster by raw AccessionNumber-vs-acc_num overlap -- it never mapped through
to anon_acc/ct_id space, and never filtered to our final 14,549-scan modeling
cohort. This script does the full join properly, against our ONE actual
modeling cohort (14,549 scans) -- not the earlier, looser precursor ID
tables (16,062- and 17,200-scan lookup tables) upstream of it, which were
only used once to cross-check where an earlier remembered count came from
and are not part of this pipeline.

Join path (verified: ct_id == anon_acc, NOT acc_num):
  roster.AccessionNumber  ==  {d2,d3}.acc_num   (raw hospital accession space)
  {d2,d3}.anon_acc / ct_id                       (our ct_id space)

Inputs (all local):
  roster  get_alicia/LCSP Completed Scans(CY2016-2026TD) LHS data share.csv   36,447 rows
  d2      tangerine_data/meta/sybil_data_17200_with_mrn_anonacc.csv          17,200 rows (anon_acc, pat_mrn_id, acc_num, proc_name, result_time)
  d3      tangerine_data/meta/lungct_with_mrn_anonacc.csv                    16,062 rows (ct_id, ..., acc_num, anon_acc)
  final   .../res_20260801/predictions/cohort_base_14549.csv                 14,549 rows (our modeling cohort)
  (d2 and d3 are only used here as acc_num->ct_id lookup tables, not as cohorts of their own)

Output -- written to the SMB share, /Volumes/For_Hua/alicia_intersection/:
  alicia_intersection_ct_ids.csv   7,985 rows, our final cohort's own schema
                                    (ct_id, pat_id, ct_date, first_lung_ca_date, last_enc_date,
                                    cancer, time_at_event, time_ct_to_last_event_or_followup, fold)
                                    -- same name and byte-identical ct_id content to this project's
                                    own alicia_intersection/alicia_intersection_ct_ids.csv (checked below).
"""
from pathlib import Path

import pandas as pd

TANGERINE_DATA = Path("/Volumes/hua_mac/research/aris/deeplearning/tangerine/tangerine_data")
HERE = Path(__file__).parent
RESULT = HERE.parent

ROSTER_CSV = HERE / "LCSP Completed Scans(CY2016-2026TD) LHS data share.csv"
D2_CSV = TANGERINE_DATA / "meta" / "sybil_data_17200_with_mrn_anonacc.csv"
D3_CSV = TANGERINE_DATA / "meta" / "lungct_with_mrn_anonacc.csv"
FINAL_COHORT_CSV = RESULT / "predictions" / "cohort_base_14549.csv"

SMB_OUT = Path("/Volumes/For_Hua/alicia_intersection")


def main():
    if not SMB_OUT.parent.is_dir():
        raise SystemExit(f"SMB share not mounted: {SMB_OUT.parent}")
    SMB_OUT.mkdir(exist_ok=True)

    roster = pd.read_csv(ROSTER_CSV, dtype=str, low_memory=False)
    d2 = pd.read_csv(D2_CSV, dtype=str)
    d3 = pd.read_csv(D3_CSV, dtype=str, low_memory=False)
    final = pd.read_csv(FINAL_COHORT_CSV, dtype=str)

    dupe_acc = roster["AccessionNumber"].duplicated().sum()
    print(f"roster: {len(roster)} rows, {roster['AccessionNumber'].nunique()} unique "
          f"AccessionNumber ({dupe_acc} duplicated)")

    acc_roster = set(roster["AccessionNumber"].dropna())
    final_ct_ids = set(final["ct_id"])

    # map roster accessions -> ct_id space via BOTH lookup tables, union them, then
    # keep only ct_ids that are actually in our final 14,549-scan modeling cohort
    m2_ct_ids = set(d2.loc[d2["acc_num"].isin(acc_roster), "anon_acc"].dropna())
    m3_ct_ids = set(d3.loc[d3["acc_num"].isin(acc_roster), "ct_id"].dropna())
    ct_ids_final = (m2_ct_ids | m3_ct_ids) & final_ct_ids

    out = final[final["ct_id"].isin(ct_ids_final)].copy()
    out_path = SMB_OUT / "alicia_intersection_ct_ids.csv"
    out.to_csv(out_path, index=False)
    print(f"-> {out_path}  {len(out)} rows (of our final 14,549-scan modeling cohort)")

    local_file = RESULT / "alicia_intersection" / "alicia_intersection_ct_ids.csv"
    if local_file.is_file():
        local_ids = set(pd.read_csv(local_file, dtype=str)["ct_id"])
        agree = local_ids == set(out["ct_id"])
        print(f"   matches local alicia_intersection_ct_ids.csv exactly: {agree} "
              f"(local n={len(local_ids)}, here n={len(out)})")


if __name__ == "__main__":
    main()
