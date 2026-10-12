"""
Sybil predict-only (V1) — NO training.

Runs a pretrained Sybil checkpoint on the SAME split (sybil-format data),
calibrates per-year on val, evaluates on test, and saves predictions in the
shared format (test_results.json, test_predictions.csv) so it's directly
comparable to the retrained Sybil (V2) and TANGERINE.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sybil_dataset import create_sybil_dataloaders
from sybil_survival_model import SybilSurvivalModel, count_parameters
from finetune_sybil_survival import SybilTrainer
from finetune_tangerine_survival import set_seed


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tr, va, te = create_sybil_dataloaders(args.dataset_dir, args.images_dir,
                                          args.batch_size, args.num_workers, augment_train=False)
    model = SybilSurvivalModel(args.sybil_checkpoint, freeze_encoder=True)
    count_parameters(model)

    # Reuse trainer's validate/calibrate machinery with the pretrained (untrained) model
    trainer = SybilTrainer(model, tr, va, te, device, args.output_dir,
                           warmup_epochs=0, use_amp=args.use_amp)

    print("\nPREDICT-ONLY: calibrate on val, evaluate on test (no fine-tuning)")
    vm = trainer.validate(va, 'Val(cal)')
    cals = trainer._calibrate(vm)
    m = trainer.validate(te, 'Test')
    cal_probs = trainer._apply_cal(m['probs'], cals)

    from sklearn.metrics import roc_auc_score
    cal_aucs = []
    for t in range(6):
        mask = ((m['cancer']==1)&(m['time']<=t)) | ((m['cancer']==0)&(m['time']>=t))
        labels = ((m['cancer']==1)&(m['time']<=t)).astype(int)
        yt, yp = labels[mask], cal_probs[mask, t]
        cal_aucs.append(float('nan') if (yt.sum()==0 or yt.sum()==len(yt)) else roc_auc_score(yt, yp))
    valid = [a for a in cal_aucs if not np.isnan(a)]
    results = {'mode': 'predict_only',
               'test_avg_auc_raw': float(m['avg_auc']),
               'test_year_aucs_raw': [None if np.isnan(a) else float(a) for a in m['year_aucs']],
               'test_avg_auc_cal': float(np.mean(valid)) if valid else 0.0,
               'test_year_aucs_cal': [None if np.isnan(a) else float(a) for a in cal_aucs]}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out/'test_results.json', 'w'), indent=2)

    def prediction_frame(metrics, cals, split):
        cal = trainer._apply_cal(metrics['probs'], cals)
        df = pd.DataFrame({
            'split': split,
            'patient_id': metrics['patient_ids'],
            'cancer': metrics['cancer'].tolist(),
            'time_at_event': metrics['time'].tolist(),
        })
        for t in range(6):
            df[f'pred_{t+1}'] = cal[:, t]
            df[f'pred_{t+1}_raw'] = metrics['probs'][:, t]
        return df

    # The training loader uses a weighted sampler and drop_last=True. Rebuild a
    # deterministic loader so every training CT is predicted exactly once.
    train_pred_loader = DataLoader(
        tr.dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    train_m = trainer.validate(train_pred_loader, 'Train(pred)')

    train_pred = prediction_frame(train_m, cals, 'train')
    val_pred = prediction_frame(vm, cals, 'val')
    test_pred = prediction_frame(m, cals, 'test')

    train_pred.to_csv(out/'train_predictions.csv', index=False)
    val_pred.to_csv(out/'val_predictions.csv', index=False)
    test_pred.to_csv(out/'test_predictions.csv', index=False)
    pd.concat([train_pred, val_pred, test_pred], ignore_index=True).to_csv(
        out/'all_predictions.csv', index=False
    )

    print(f"\nPredict-only raw avg-AUC={results['test_avg_auc_raw']:.4f}  "
          f"cal avg-AUC={results['test_avg_auc_cal']:.4f}")
    print("Saved test/train/val/all prediction CSVs")
    print(f"Saved to {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir', required=True)
    p.add_argument('--images_dir', required=True, help='images_3d_tan_syb/sybil')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--sybil_checkpoint', required=True)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--seed', type=int, default=42)
    main(p.parse_args())
