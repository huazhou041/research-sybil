"""
Sybil model wrapper for 6-year survival fine-tuning + bottleneck extraction.

Wraps the official SybilNet (R3D-18 encoder + MultiAttentionPool +
Cumulative_Probability_Layer). Sybil is already survival-native, so we simply:
  - load a pretrained Sybil checkpoint
  - expose forward(x) -> [B, 6] logits  (for retraining)
  - expose get_hidden(x) -> [B, 512] bottleneck  (for ensemble fusion, method B)
  - freeze/unfreeze the image encoder for warmup-then-finetune
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Sybil package lives in ./sybil (copied into this dir)
_sybil_path = Path(__file__).parent / 'sybil'
sys.path.insert(0, str(_sybil_path))

from sybil.models.sybil import SybilNet

HIDDEN_DIM = 512  # Sybil bottleneck dimension


class SybilSurvivalModel(nn.Module):
    def __init__(self, checkpoint_path: str, freeze_encoder: bool = True):
        super().__init__()
        print(f"\n{'='*60}\nLOADING PRETRAINED SYBIL\n{'='*60}")
        self.sybil = SybilNet.load(checkpoint_path)
        print(f"  Loaded Sybil from: {checkpoint_path}")
        print(f"  hidden_dim: {self.sybil.hidden_dim}")
        if freeze_encoder:
            self._freeze_encoder()
            print(f"  Encoder (image_encoder) frozen for warmup")
        print(f"{'='*60}\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, T, H, W] -> [B, 6] cumulative-hazard logits."""
        out = self.sybil(x)
        return out['logit']

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-d bottleneck (post-pool, post-relu) for fusion."""
        a = self.sybil.image_encoder(x)
        pool_output = self.sybil.pool(a)
        hidden = self.sybil.relu(pool_output['hidden'])
        return hidden  # [B, 512]

    # head = the cumulative probability layer (+ pooling) trained during warmup
    @property
    def head(self):
        return self.sybil.prob_of_failure_layer

    def _freeze_encoder(self):
        for p in self.sybil.image_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.sybil.image_encoder.parameters():
            p.requires_grad = True
        print("  Sybil image_encoder unfrozen")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*70}\nSYBIL MODEL PARAMETERS\n{'='*70}")
    print(f"Total: {total:,}  Trainable: {trainable:,}  Frozen: {total-trainable:,}")
    print(f"{'='*70}\n")
    return total, trainable
