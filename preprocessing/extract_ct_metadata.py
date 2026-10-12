"""
Extract ALL DICOM metadata from every CT scan into one CSV.

For each CT folder we dump every top-level DICOM tag from a representative
file of the SERIES THAT WAS ACTUALLY SELECTED by the TANGERINE/Sybil
preprocessing (series_selection.select_best_series). The selected series UID
is read from the existing tangerine preprocessing logs, so we do NOT pay the
~90s SimpleITK GetGDCMSeriesIDs scan per folder — we just pydicom-scan the
folder (~3-4s) for the file whose SeriesInstanceUID matches.

Columns added on top of the raw DICOM tags:
    folder_name        : the data-root subfolder (== AccessionNumber here)
    mrn                : PatientID (the patient MRN)
    in_tangerine       : 1 if <name>.nii.gz exists in the tangerine output
    in_sybil           : 1 if <name>.nii.gz exists in the sybil output
    dataset_membership : both | tangerine_only | sybil_only | none
    selected_series_uid: SeriesInstanceUID used for preprocessing
    rep_file           : representative DICOM file the metadata was read from
    n_files_in_folder  : total .dcm files in the folder

Resumable SLURM array job; merge step concatenates per-job JSON -> CSV.

Usage (single job):
    python extract_ct_metadata.py --data_root ... --tangerine_dir ... \
        --sybil_dir ... --output_root ... --job_id 0 --total_jobs 20
Merge:
    python extract_ct_metadata.py --output_root ... --total_jobs 20 --merge
"""
import os
import sys
import json
import glob
import argparse
import tempfile
from pathlib import Path
from datetime import datetime

import pydicom
from pydicom.tag import Tag

PIXEL_DATA = Tag(0x7FE0, 0x0010)
BINARY_VRS = {"OB", "OW", "OF", "OD", "OL", "UN"}


def _atomic_json_dump(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rows, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_selected_uids(tangerine_dir):
    """folder_name -> selected SeriesInstanceUID (from tangerine logs)."""
    uid = {}
    logs = sorted(glob.glob(str(Path(tangerine_dir) / "preprocessing_log_job_*.json")))
    if not logs:
        merged = Path(tangerine_dir) / "preprocessing_log.json"
        if merged.exists():
            logs = [str(merged)]
    for lf in logs:
        try:
            for r in json.load(open(lf)):
                name = r.get("folder_name")
                if name and r.get("SeriesInstanceUID"):
                    uid[name] = str(r["SeriesInstanceUID"])
        except Exception:
            pass
    return uid


def list_outputs(out_dir):
    """Set of folder_names that have a <name>.nii.gz in out_dir."""
    names = set()
    p = Path(out_dir)
    if not p.is_dir():
        return names
    with os.scandir(p) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".nii.gz"):
                names.add(e.name[:-len(".nii.gz")])
    return names


def scan_folders(data_root):
    folders = []
    with os.scandir(data_root) as it:
        for e in it:
            if e.is_dir(follow_symlinks=True):
                f = Path(e.path)
                # cheap existence check without globbing thousands of names
                try:
                    has_dcm = any(c.name.endswith(".dcm") for c in os.scandir(f))
                except Exception:
                    has_dcm = False
                if has_dcm:
                    folders.append(f)
    return sorted(folders, key=lambda p: p.name)


def find_rep_file(folder, target_uid):
    """First .dcm whose SeriesInstanceUID == target_uid; else first readable."""
    files = sorted(Path(folder).glob("*.dcm"))
    fallback = None
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True,
                                 specific_tags=["SeriesInstanceUID"], force=True)
        except Exception:
            continue
        if fallback is None:
            fallback = p
        if target_uid and str(getattr(ds, "SeriesInstanceUID", "")) == target_uid:
            return p, len(files)
    return fallback, len(files)


def dump_all_tags(dcm_path):
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    row = {}
    for elem in ds:
        if elem.tag == PIXEL_DATA:
            continue
        name = elem.keyword or f"({elem.tag.group:04X},{elem.tag.element:04X})"
        vr = elem.VR
        try:
            if vr in BINARY_VRS:
                try:
                    ln = len(elem.value)
                except Exception:
                    ln = "?"
                val = f"<binary {vr} {ln}>"
            elif vr == "SQ":
                val = str(elem.value)[:500]
            else:
                val = str(elem.value)
        except Exception as e:
            val = f"<unreadable: {e}>"
        if len(val) > 1000:
            val = val[:1000]
        row[name] = val
    return row


def process_folder(folder, target_uid, in_tan, in_syb):
    name = Path(folder).name
    rep, n_files = find_rep_file(folder, target_uid)
    base = {
        "folder_name": name,
        "in_tangerine": int(in_tan),
        "in_sybil": int(in_syb),
        "dataset_membership": ("both" if in_tan and in_syb else
                               "tangerine_only" if in_tan else
                               "sybil_only" if in_syb else "none"),
        "selected_series_uid": target_uid or "",
        "rep_file": str(rep) if rep else "",
        "n_files_in_folder": n_files,
    }
    if rep is None:
        base["mrn"] = ""
        base["_error"] = "no readable dicom"
        return base
    try:
        tags = dump_all_tags(rep)
    except Exception as e:
        base["mrn"] = ""
        base["_error"] = f"dump failed: {e}"
        return base
    base["mrn"] = tags.get("PatientID", "")
    base.update(tags)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root")
    ap.add_argument("--tangerine_dir")
    ap.add_argument("--sybil_dir")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--job_id", type=int, default=0)
    ap.add_argument("--total_jobs", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.merge:
        import pandas as pd
        rows = []
        for j in range(args.total_jobs):
            jf = out_root / f"metadata_job_{j}.json"
            if jf.exists():
                rows.extend(json.load(open(jf)))
        if not rows:
            print("[WARN] no per-job json found"); return
        df = pd.DataFrame(rows)
        lead = ["folder_name", "mrn", "PatientID", "PatientName",
                "dataset_membership", "in_tangerine", "in_sybil",
                "AccessionNumber", "StudyInstanceUID", "selected_series_uid",
                "SeriesInstanceUID", "SeriesDescription", "n_files_in_folder",
                "rep_file"]
        cols = [c for c in lead if c in df.columns] + \
               sorted(c for c in df.columns if c not in lead)
        df = df[cols]
        csv_path = out_root / "ct_metadata_all.csv"
        df.to_csv(csv_path, index=False)
        print(f"[MERGE] {len(df)} scans, {len(df.columns)} columns -> {csv_path}")
        print(f"[MERGE] membership: \n{df['dataset_membership'].value_counts()}")
        return

    for req in ("data_root", "tangerine_dir", "sybil_dir"):
        if not getattr(args, req):
            ap.error(f"--{req} required for extraction")

    uid_map = load_selected_uids(args.tangerine_dir)
    tan_set = list_outputs(args.tangerine_dir)
    syb_set = list_outputs(args.sybil_dir)
    print(f"[INFO] selected-UID map: {len(uid_map)}; "
          f"tangerine outputs: {len(tan_set)}; sybil outputs: {len(syb_set)}",
          flush=True)

    folders = scan_folders(args.data_root)
    print(f"[INFO] {len(folders)} dicom folders", flush=True)

    per = len(folders) // args.total_jobs
    s = args.job_id * per
    e = s + per if args.job_id < args.total_jobs - 1 else len(folders)
    job = folders[s:e]
    print(f"[JOB {args.job_id}/{args.total_jobs}] folders {s}..{e-1} (n={len(job)})", flush=True)

    job_file = out_root / f"metadata_job_{args.job_id}.json"
    done = {}
    if job_file.exists():
        try:
            for r in json.load(open(job_file)):
                done[r["folder_name"]] = r
            print(f"[INFO] resume: {len(done)} already extracted", flush=True)
        except Exception:
            done = {}

    results = list(done.values())
    t0 = datetime.now()
    for i, f in enumerate(job):
        name = f.name
        if name in done:
            continue
        rec = process_folder(f, uid_map.get(name),
                             name in tan_set, name in syb_set)
        results.append(rec)
        if (i + 1) % 25 == 0 or i == 0:
            dt = (datetime.now() - t0).total_seconds()
            print(f"[JOB {args.job_id}] {i+1}/{len(job)} ({dt:.0f}s)", flush=True)
            _atomic_json_dump(results, job_file)
    _atomic_json_dump(results, job_file)
    err = sum(1 for r in results if r.get("_error"))
    print(f"[JOB {args.job_id}] done: {len(results)} rows, {err} errors", flush=True)


if __name__ == "__main__":
    main()
