#!/usr/bin/env python3
"""Sybil predict-only ensemble on the current split.

Loads the five official Sybil checkpoints one at a time, averages raw
probabilities, calibrates on validation, and writes train/val/test/all
prediction CSVs matching predict_sybil.py.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from finetune_tangerine_survival import compute_auc_at_year, set_seed
from sybil_dataset import create_sybil_dataloaders
from sybil_survival_model import SybilSurvivalModel

MAX_FOLLOWUP = 6


def validate_model(model, loader, device, split, use_amp=True):
    model.eval()
    probs_all, cancer_all, time_all, pids = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=split):
            vol = batch["volume"].to(device)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(vol)
            else:
                logits = model(vol)
            probs_all.extend(torch.sigmoid(logits).cpu().numpy())
            cancer_all.extend(batch["cancer"].numpy())
            time_all.extend(batch["time_at_event"].numpy())
            pids.extend(batch["patient_id"])
    probs = np.array(probs_all)
    cancer = np.array(cancer_all)
    time = np.array(time_all)
    year_aucs = [compute_auc_at_year(probs, cancer, time, t) for t in range(MAX_FOLLOWUP)]
    valid = [a for a in year_aucs if not np.isnan(a)]
    return {
        "avg_auc": float(np.mean(valid)) if valid else 0.0,
        "year_aucs": year_aucs,
        "probs": probs,
        "cancer": cancer,
        "time": time,
        "patient_ids": pids,
    }


def add_metric(acc, metrics):
    if acc is None:
        return {
            "probs": metrics["probs"].copy(),
            "cancer": metrics["cancer"].copy(),
            "time": metrics["time"].copy(),
            "patient_ids": list(metrics["patient_ids"]),
        }
    if acc["patient_ids"] != metrics["patient_ids"]:
        raise RuntimeError("Prediction order changed between checkpoints.")
    if not np.array_equal(acc["cancer"], metrics["cancer"]) or not np.allclose(acc["time"], metrics["time"]):
        raise RuntimeError("Labels changed between checkpoints.")
    acc["probs"] += metrics["probs"]
    return acc


def finalize_metric(acc, n_models):
    acc["probs"] = acc["probs"] / float(n_models)
    acc["year_aucs"] = [compute_auc_at_year(acc["probs"], acc["cancer"], acc["time"], t) for t in range(MAX_FOLLOWUP)]
    valid = [a for a in acc["year_aucs"] if not np.isnan(a)]
    acc["avg_auc"] = float(np.mean(valid)) if valid else 0.0
    return acc


def calibrate(vm, output_dir):
    import pickle

    cals = {}
    for t in range(MAX_FOLLOWUP):
        mask = ((vm["cancer"] == 1) & (vm["time"] <= t)) | ((vm["cancer"] == 0) & (vm["time"] >= t))
        labels = ((vm["cancer"] == 1) & (vm["time"] <= t)).astype(int)
        yt, yr = labels[mask], vm["probs"][mask, t].reshape(-1, 1)
        if yt.sum() == 0 or yt.sum() == len(yt):
            cals[f"Year{t+1}"] = None
            continue
        c = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5, method="isotonic")
        c.fit(yr, yt)
        cals[f"Year{t+1}"] = c
    with open(Path(output_dir) / "calibrator.pkl", "wb") as f:
        pickle.dump(cals, f)
    return cals


def apply_cal(probs, cals):
    out = probs.copy()
    for t in range(MAX_FOLLOWUP):
        c = cals.get(f"Year{t+1}")
        if c is not None:
            out[:, t] = c.predict_proba(probs[:, t].reshape(-1, 1))[:, 1]
    return np.maximum.accumulate(out, axis=1)


def calibrated_aucs(metrics, cal_probs):
    aucs = []
    for t in range(MAX_FOLLOWUP):
        mask = ((metrics["cancer"] == 1) & (metrics["time"] <= t)) | ((metrics["cancer"] == 0) & (metrics["time"] >= t))
        labels = ((metrics["cancer"] == 1) & (metrics["time"] <= t)).astype(int)
        yt, yp = labels[mask], cal_probs[mask, t]
        aucs.append(float("nan") if yt.sum() == 0 or yt.sum() == len(yt) else roc_auc_score(yt, yp))
    return aucs


def prediction_frame(metrics, cals, split):
    cal = apply_cal(metrics["probs"], cals)
    df = pd.DataFrame(
        {
            "split": split,
            "patient_id": metrics["patient_ids"],
            "cancer": metrics["cancer"].tolist(),
            "time_at_event": metrics["time"].tolist(),
        }
    )
    for t in range(MAX_FOLLOWUP):
        df[f"pred_{t+1}"] = cal[:, t]
        df[f"pred_{t+1}_raw"] = metrics["probs"][:, t]
    return df


def main(args):
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr, va, te = create_sybil_dataloaders(
        args.dataset_dir, args.images_dir, args.batch_size, args.num_workers, augment_train=False
    )
    train_pred_loader = DataLoader(
        tr.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    train_acc = val_acc = test_acc = None
    checkpoints = [p.strip() for p in args.sybil_checkpoints.split(",") if p.strip()]
    for idx, ckpt in enumerate(checkpoints, start=1):
        print(f"\n=== Ensemble member {idx}/{len(checkpoints)}: {ckpt} ===")
        model = SybilSurvivalModel(ckpt, freeze_encoder=True).to(device)
        train_acc = add_metric(train_acc, validate_model(model, train_pred_loader, device, f"Train member {idx}", args.use_amp))
        val_acc = add_metric(val_acc, validate_model(model, va, device, f"Val member {idx}", args.use_amp))
        test_acc = add_metric(test_acc, validate_model(model, te, device, f"Test member {idx}", args.use_amp))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_m = finalize_metric(train_acc, len(checkpoints))
    val_m = finalize_metric(val_acc, len(checkpoints))
    test_m = finalize_metric(test_acc, len(checkpoints))

    cals = calibrate(val_m, out)
    test_cal = apply_cal(test_m["probs"], cals)
    cal_aucs = calibrated_aucs(test_m, test_cal)
    valid_cal = [a for a in cal_aucs if not np.isnan(a)]

    results = {
        "mode": "predict_only_ensemble",
        "n_models": len(checkpoints),
        "checkpoints": checkpoints,
        "test_avg_auc_raw": float(test_m["avg_auc"]),
        "test_year_aucs_raw": [None if np.isnan(a) else float(a) for a in test_m["year_aucs"]],
        "test_avg_auc_cal": float(np.mean(valid_cal)) if valid_cal else 0.0,
        "test_year_aucs_cal": [None if np.isnan(a) else float(a) for a in cal_aucs],
    }
    with open(out / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    train_pred = prediction_frame(train_m, cals, "train")
    val_pred = prediction_frame(val_m, cals, "val")
    test_pred = prediction_frame(test_m, cals, "test")
    train_pred.to_csv(out / "train_predictions.csv", index=False)
    val_pred.to_csv(out / "val_predictions.csv", index=False)
    test_pred.to_csv(out / "test_predictions.csv", index=False)
    pd.concat([train_pred, val_pred, test_pred], ignore_index=True).to_csv(out / "all_predictions.csv", index=False)

    print(
        f"\nEnsemble predict-only raw avg-AUC={results['test_avg_auc_raw']:.4f} "
        f"cal avg-AUC={results['test_avg_auc_cal']:.4f}"
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sybil_checkpoints", required=True, help="comma-separated .ckpt paths")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
