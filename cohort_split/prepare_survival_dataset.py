"""
Prepare dataset for 6-year survival prediction with TANGERINE.

Filters time_at_event < 0, computes y_seq / y_mask (Sybil-style cumulative
labels), matches to preprocessed images, and creates stratified splits.

Label encoding (Sybil-style):
    diagnosis at year 3  ->  y_seq  = [0,0,1,1,1,1]   (cumulative cancer status)
                             y_mask = [1,1,1,0,0,0]   (supervise observed period)

Split: PATIENT-level stratified split (70/15/15 default) — prevents data
       leakage by keeping all of a patient's CTs in the same split.
       (Matches the v2 prepare_survival_dataset.py.)

Cohort reconcile is INTEGRATED: pass --intersect_with to keep only CTs present
in ALL preprocessed sets (e.g. tangerine AND sybil), so both models train and
ensemble on an identical cohort. Writes common_ct_ids.txt to the output dir.

Usage (single set):
    python prepare_survival_dataset.py \
        --metadata_csv /path/to/lungct_with_mrn_anonacc.csv \
        --images_dir   /path/to/images_3d_tan_syb/tangerine \
        --output_dir   ./dataset_splits \
        --max_followup 6 --seed 42

Usage (common cohort across tangerine + sybil):
    python prepare_survival_dataset.py \
        --metadata_csv  /path/to/lungct_with_mrn_anonacc.csv \
        --images_dir    /path/to/images_3d_tan_syb/tangerine \
        --intersect_with /path/to/images_3d_tan_syb/sybil \
        --output_dir    ./dataset_splits --seed 42
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

MAX_FOLLOWUP = 6


def make_survival_labels(cancer: int, time_years: float, max_followup: int):
    """Return y_seq [T], y_mask [T], time_at_event for one CT scan."""
    y_seq = np.zeros(max_followup, dtype=np.float32)
    y_mask = np.zeros(max_followup, dtype=np.float32)
    time_at_event = int(min(time_years, max_followup - 1))  # floor to year index

    if cancer == 1:
        y_seq[time_at_event:] = 1.0          # 1 from diagnosis year onward
    # mask: supervise up to time_at_event + 1 years
    y_mask[:time_at_event + 1] = 1.0
    return y_seq, y_mask, time_at_event


def prepare(args):
    print(f"\n{'='*65}")
    print(f"SURVIVAL DATASET PREPARATION (TANGERINE)")
    print(f"{'='*65}\n")

    df = pd.read_csv(args.metadata_csv)
    if df.columns[0].startswith('Unnamed'):
        df = df.drop(columns=[df.columns[0]])
    print(f"Loaded {len(df)} records from metadata CSV")

    # ── Identify ID column ──────────────────────────────────────────────
    for col in ['ct_id', 'AnonAcc', 'MRN']:
        if col in df.columns:
            id_col = col
            break
    else:
        raise ValueError("No ID column found (ct_id / AnonAcc / MRN)")
    print(f"ID column: {id_col}")

    # ── Filter time_at_event < 0 ────────────────────────────────────────
    before = len(df)
    df = df[df['time_ct_to_last_event_or_followup'] >= 0].copy()
    print(f"After removing time<0: {len(df)} records (removed {before - len(df)})")
    print(f"  Cancer positive: {int(df['cancer'].sum())}")
    print(f"  Cancer negative: {int((df['cancer']==0).sum())}")

    # ── Match against COMMON cohort across preprocessed sets ────────────
    # (Integrated reconcile: a CT is kept only if it exists in --images_dir
    #  AND every dir in --intersect_with. Sybil drops all-thick CTs, so the
    #  intersection is the shared cohort both models can train/ensemble on.)
    def _ct_ids(d):
        return {f.name[:-7] for f in Path(d).glob('*.nii.gz')}  # strip '.nii.gz'

    images_path = Path(args.images_dir)
    primary = _ct_ids(images_path)
    print(f"Preprocessed sets:")
    print(f"  {images_path.name:12s}: {len(primary)} CTs   ({images_path})")
    common = set(primary)
    set_sizes = {images_path.name: len(primary)}
    for extra in (args.intersect_with or []):
        ep = Path(extra)
        eset = _ct_ids(ep)
        set_sizes[ep.name] = len(eset)
        print(f"  {ep.name:12s}: {len(eset)} CTs   ({ep})")
        common &= eset
    if args.intersect_with:
        print(f"  {'COMMON':12s}: {len(common)} CTs (intersection of all sets)")
        only_primary = len(primary - common)
        print(f"  (only in {images_path.name}, dropped: {only_primary})")

    df['image_filename'] = df[id_col].astype(str) + '.nii.gz'
    df = df[df[id_col].astype(str).isin(common)].copy()
    print(f"After cohort matching: {len(df)} records")

    # ── Optional slice-thickness filter (Sybil-style ablation) ──────────
    # TANGERINE original applies NO thickness filter; default here is OFF.
    # If --max_slice_thickness is set AND a preprocessing_log is given,
    # merge per-scan z-spacing and drop scans above the threshold.
    if args.max_slice_thickness is not None:
        if not args.preprocessing_log or not Path(args.preprocessing_log).exists():
            raise ValueError("--max_slice_thickness requires a valid --preprocessing_log "
                             "(preprocessing_log.csv from preprocess_tangerine3d.py)")
        log = pd.read_csv(args.preprocessing_log)
        # Prefer measured z-spacing; fall back to SliceThickness tag
        zcol = 'OriginalSpacingZ' if 'OriginalSpacingZ' in log.columns else 'SliceThickness'
        log = log[['folder_name', zcol]].copy()
        log['folder_name'] = log['folder_name'].astype(str)
        thick = dict(zip(log['folder_name'], pd.to_numeric(log[zcol], errors='coerce')))
        before_t = len(df)
        df['_zthick'] = df[id_col].astype(str).map(thick)
        # Keep scans with known thickness within range (drop unknown to be safe)
        df = df[df['_zthick'].notna() & (df['_zthick'] > 0) &
                (df['_zthick'] <= args.max_slice_thickness)].copy()
        df = df.drop(columns=['_zthick'])
        print(f"After thickness filter (<= {args.max_slice_thickness}mm): "
              f"{len(df)} records (removed {before_t - len(df)})")
    else:
        print("Thickness filter: OFF (matches TANGERINE original)")

    # ── Compute survival labels ─────────────────────────────────────────
    y_seqs, y_masks, tae_list = [], [], []
    for _, row in df.iterrows():
        ys, ym, tae = make_survival_labels(
            int(row['cancer']),
            float(row['time_ct_to_last_event_or_followup']),
            args.max_followup)
        y_seqs.append(ys); y_masks.append(ym); tae_list.append(tae)

    for t in range(args.max_followup):
        df[f'y_seq_{t}'] = [s[t] for s in y_seqs]
        df[f'y_mask_{t}'] = [m[t] for m in y_masks]
    df['time_at_event'] = tae_list

    # ── Evaluable counts per year ───────────────────────────────────────
    print(f"\nEvaluable per year cutoff:")
    for yr in range(1, args.max_followup + 1):
        vp = int(((df['cancer']==1) & (df['time_ct_to_last_event_or_followup'] <= yr)).sum())
        vn = int(((df['cancer']==0) & (df['time_ct_to_last_event_or_followup'] >= yr)).sum())
        print(f"  Year {yr}: pos={vp:4d}  neg={vn:5d}  total={vp+vn:5d}")

    # ── PATIENT-level stratified split (prevent data leakage) ───────────
    print(f"\n{'-'*65}")
    print("PATIENT-LEVEL SPLIT (no same-patient leakage across splits)")
    print(f"{'-'*65}")

    # One row per patient. Aggregate:
    #   cancer     = patient cancer status (first)
    #   num_cts    = number of CT scans for this patient
    #   event_year = earliest diagnosis horizon (min time_at_event) for positives
    #                (used for finer stratification so cancer TIMING is balanced
    #                 across splits, not just cancer presence)
    patient_df = df.groupby('PatientID').agg(
        cancer=('cancer', 'first'),
        num_cts=(id_col, 'count'),
        event_year=('time_at_event', 'min'),
    ).reset_index()
    print(f"Unique patients: {len(patient_df)}  "
          f"(cancer+={int(patient_df['cancer'].sum())}  "
          f"cancer-={int((patient_df['cancer']==0).sum())})")

    # ── Build stratification key: cancer x event-year bin ───────────────
    # Negatives -> single 'neg' bin. Positives -> 'pos_y{event_year}'.
    # Rare bins (< min_bin patients) are merged into 'pos_other' so
    # train_test_split always has >=2 members per class.
    def _key(row):
        if int(row['cancer']) == 0:
            return 'neg'
        return f"pos_y{int(row['event_year'])}"
    patient_df['strat'] = patient_df.apply(_key, axis=1)

    min_bin = 6  # need enough to survive a 70/15/15 split
    counts = patient_df['strat'].value_counts()
    rare = set(counts[counts < min_bin].index)
    if rare:
        patient_df['strat'] = patient_df['strat'].apply(
            lambda k: ('pos_other' if (k in rare and k != 'neg') else k))
        # if 'pos_other' itself is too small, fold it into 'neg'-safe cancer-only
        if (patient_df['strat'] == 'pos_other').sum() < min_bin:
            patient_df.loc[patient_df['strat'] == 'pos_other', 'strat'] = \
                patient_df.loc[patient_df['strat'] == 'pos_other', 'cancer'].map({1: 'pos_y0', 0: 'neg'})

    print("Stratification bins (patient counts):")
    for k, v in patient_df['strat'].value_counts().sort_index().items():
        print(f"  {k:12s}: {v}")

    def _split_stratified(frame, test_size, key):
        """train_test_split with graceful fallback to cancer-only stratify."""
        try:
            return train_test_split(frame, test_size=test_size,
                                    stratify=frame[key], random_state=args.seed)
        except ValueError:
            print(f"  [warn] fine stratify failed (rare bin) — falling back to cancer-only")
            return train_test_split(frame, test_size=test_size,
                                    stratify=frame['cancer'], random_state=args.seed)

    train_pats, temp_pats = _split_stratified(
        patient_df, args.val_ratio + args.test_ratio, 'strat')
    val_ratio_adj = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_pats, test_pats = _split_stratified(
        temp_pats, 1 - val_ratio_adj, 'strat')

    print(f"\nPatient splits:")
    for name, pats in [('Train', train_pats), ('Val', val_pats), ('Test', test_pats)]:
        n_pos = int(pats['cancer'].sum())
        print(f"  {name}: {len(pats):4d} patients  (cancer+={n_pos:3d}  cancer-={len(pats)-n_pos:4d})")

    # Map all CTs of each patient to that patient's split
    train_ids = set(train_pats['PatientID'])
    val_ids = set(val_pats['PatientID'])
    test_ids = set(test_pats['PatientID'])
    train_df = df[df['PatientID'].isin(train_ids)].copy()
    val_df = df[df['PatientID'].isin(val_ids)].copy()
    test_df = df[df['PatientID'].isin(test_ids)].copy()

    print(f"\nCT-level counts (after patient assignment):")
    for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"  {name}: {len(d):5d} CTs  (cancer={int(d['cancer'].sum())})")

    # ── Per-year POSITIVE / NEGATIVE CT balance report ──────────────────
    # Uses the same Sybil masking as AUC computation so the counts reflect
    # exactly the cases each year's AUC is evaluated on. Watch for years with
    # very few positives in Val/Test (-> noisy / nan AUC).
    print(f"\nPer-year evaluable POSITIVE CTs per split "
          f"(pos = cancer=1 & time<=yr ; in parentheses: negatives):")
    print(f"  {'Year':<6}{'Train pos(neg)':>20}{'Val pos(neg)':>18}{'Test pos(neg)':>18}")
    for yr in range(args.max_followup):
        row = f"  {yr+1:<6}"
        for d in (train_df, val_df, test_df):
            c = d['cancer'].values
            t = d['time_at_event'].values
            pos = int(((c == 1) & (t <= yr)).sum())
            neg = int(((c == 0) & (t >= yr)).sum())
            row += f"{f'{pos}({neg})':>{20 if d is train_df else 18}}"
        print(row)

    # Leakage verification
    tv = len(train_ids & val_ids); ts = len(train_ids & test_ids); vs = len(val_ids & test_ids)
    print(f"\nLeakage check — Train-Val:{tv}  Train-Test:{ts}  Val-Test:{vs}")
    if tv == 0 and ts == 0 and vs == 0:
        print("  ✓ No patient overlap — splits valid")
    else:
        raise RuntimeError("✗ Patient overlap detected — data leakage!")

    # ── Save ────────────────────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / 'train.csv', index=False)
    val_df.to_csv(out / 'val.csv', index=False)
    test_df.to_csv(out / 'test.csv', index=False)

    # Record the common cohort (CTs used across all sets) for traceability
    with open(out / 'common_ct_ids.txt', 'w') as f:
        for cid in sorted(common):
            f.write(cid + '\n')

    config = {
        'max_followup': args.max_followup,
        'split_method': 'patient-level',
        'stratify_key': 'cancer_x_event_year',
        'max_slice_thickness': args.max_slice_thickness,
        'images_dir': str(images_path),
        'intersect_with': list(args.intersect_with or []),
        'preprocessed_set_sizes': set_sizes,
        'common_cohort_cts': len(common),
        'total_patients': len(patient_df),
        'total_cts': len(df),
        'cancer_pos': int(df['cancer'].sum()),
        'cancer_neg': int((df['cancer']==0).sum()),
        'train_patients': len(train_pats), 'train_cts': len(train_df),
        'val_patients': len(val_pats), 'val_cts': len(val_df),
        'test_patients': len(test_pats), 'test_cts': len(test_df),
        'seed': args.seed,
    }
    with open(out / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata_csv', required=True)
    p.add_argument('--images_dir',   required=True,
                   help='Primary preprocessed set (e.g. images_3d_tan_syb/tangerine)')
    p.add_argument('--intersect_with', nargs='*', default=None,
                   help='Additional preprocessed set(s) to intersect with, e.g. '
                        'images_3d_tan_syb/sybil. A CT is kept only if present in ALL sets '
                        '(integrated cohort reconcile for fair ensembling).')
    p.add_argument('--output_dir',   default='./dataset_splits')
    p.add_argument('--max_followup', type=int,   default=6)
    p.add_argument('--train_ratio',  type=float, default=0.70)
    p.add_argument('--val_ratio',    type=float, default=0.15)
    p.add_argument('--test_ratio',   type=float, default=0.15)
    p.add_argument('--seed',         type=int,   default=42)
    # Optional Sybil-style slice-thickness filter (default OFF = TANGERINE original)
    p.add_argument('--max_slice_thickness', type=float, default=None,
                   help='Drop scans with z-spacing > this (mm). Requires --preprocessing_log. '
                        'Default None = no filter (matches TANGERINE original). '
                        'Sybil-style ablation: 2.5')
    p.add_argument('--preprocessing_log', type=str, default=None,
                   help='Path to preprocessing_log.csv (from preprocess_tangerine3d.py) '
                        'providing per-scan OriginalSpacingZ / SliceThickness')
    prepare(p.parse_args())
