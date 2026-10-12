"""
Dataset for Sybil-format 6-year survival prediction.

Loads the Sybil-preprocessed volumes (from preprocess_sybil3d.py):
    saved 1-channel, windowed [0,255], shape (Z=200, Y=256, X=256)
and reproduces Sybil's model input at LOAD time:
    z-score norm (mean 128.1722 / std 87.1849), expand to 3 channels
    -> tensor (C=3, T=200, H=256, W=256) for r3d_18

Survival labels (y_seq / y_mask) come from the SAME split CSVs used by the
TANGERINE model, so both models train/eval on the identical cohort.
"""
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

MAX_FOLLOWUP = 6

# Sybil normalization constants (sybil args.img_mean / img_std)
SYBIL_MEAN = 128.1722
SYBIL_STD = 87.1849
NUM_CHAN = 3
NUM_IMAGES = 200  # depth (T)
IMG_HW = (256, 256)


class SybilSurvivalDataset(Dataset):

    def __init__(self, csv_file, images_dir, augment=False, mode='train'):
        self.df = pd.read_csv(csv_file)
        if self.df.columns[0].startswith('Unnamed'):
            self.df = self.df.drop(columns=[self.df.columns[0]])
        self.images_dir = Path(images_dir)
        self.augment = augment and (mode == 'train')
        self.mode = mode

        for col in ['ct_id', 'AnonAcc', 'MRN']:
            if col in self.df.columns:
                self.id_col = col
                break
        else:
            raise ValueError("No ID column found")

        print(f"[SYBIL {mode.upper()}] {len(self.df)} samples  "
              f"(cancer+={int(self.df['cancer'].sum())}  "
              f"cancer-={int((self.df['cancer']==0).sum())})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load (Z=200, Y=256, X=256) windowed 1-channel volume
        vol = sitk.GetArrayFromImage(
            sitk.ReadImage(str(self.images_dir / row['image_filename']))
        ).astype(np.float32)
        while vol.ndim > 3:
            vol = vol.squeeze()

        # Safety: enforce (200, 256, 256)
        vol = self._fit(vol)

        # Sybil z-score normalization (single channel)
        vol = (vol - SYBIL_MEAN) / SYBIL_STD

        # Light intensity augmentation (train) — keep geometry intact
        if self.augment:
            if np.random.rand() < 0.1:
                vol = vol + np.random.normal(0, 0.05, size=vol.shape).astype(np.float32)

        # Expand to 3 channels -> (C=3, T=200, H=256, W=256)
        vol = np.repeat(vol[np.newaxis], NUM_CHAN, axis=0)  # (3, T, H, W)
        volume = torch.from_numpy(np.ascontiguousarray(vol)).float()

        y_seq = np.array([row[f'y_seq_{t}'] for t in range(MAX_FOLLOWUP)], dtype=np.float32)
        y_mask = np.array([row[f'y_mask_{t}'] for t in range(MAX_FOLLOWUP)], dtype=np.float32)

        return {
            'volume':        volume,
            'y_seq':         torch.from_numpy(y_seq),
            'y_mask':        torch.from_numpy(y_mask),
            'cancer':        torch.tensor(int(row['cancer']), dtype=torch.long),
            'time_at_event': torch.tensor(float(row['time_at_event']), dtype=torch.float32),
            'patient_id':    str(row[self.id_col]),
        }

    def _fit(self, vol):
        """Enforce (T=200, H=256, W=256): center pad/crop, pad value 0 (windowed background)."""
        tT, tH, tW = NUM_IMAGES, IMG_HW[0], IMG_HW[1]
        z, h, w = vol.shape
        if z < tT or h < tH or w < tW:
            vol = np.pad(vol, ((0, max(0, tT-z)), (0, max(0, tH-h)), (0, max(0, tW-w))),
                         mode='constant', constant_values=0)
            z, h, w = vol.shape
        zs, hs, ws = (z-tT)//2, (h-tH)//2, (w-tW)//2
        return vol[zs:zs+tT, hs:hs+tH, ws:ws+tW]


def create_sybil_dataloaders(dataset_dir, images_dir, batch_size=2,
                             num_workers=8, augment_train=True):
    dpath = Path(dataset_dir)
    train_ds = SybilSurvivalDataset(dpath/'train.csv', images_dir, augment_train, 'train')
    val_ds = SybilSurvivalDataset(dpath/'val.csv', images_dir, False, 'val')
    test_ds = SybilSurvivalDataset(dpath/'test.csv', images_dir, False, 'test')

    labels = train_ds.df['cancer'].values
    class_counts = np.bincount(labels)
    sample_w = (1.0 / class_counts)[labels]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    print(f"\nWeighted sampler: neg={class_counts[0]}  pos={class_counts[1]}  "
          f"ratio={class_counts[0]/class_counts[1]:.1f}:1")

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
