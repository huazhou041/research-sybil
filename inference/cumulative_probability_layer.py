"""
Cumulative Probability Layer — adapted from Sybil (MIT License).
Standalone version with no external Sybil dependencies.
"""
import torch
import torch.nn as nn


class CumulativeProbabilityLayer(nn.Module):
    """
    Takes encoder features [B, D] and outputs cumulative cancer risk
    logits [B, T] for T yearly timepoints.

    For each year t, the output logit represents log-odds of cancer
    by year t, built as a cumulative sum of per-year hazards plus a
    base hazard, enforcing monotonically non-decreasing risk via the
    upper-triangular mask.
    """
    def __init__(self, num_features: int, max_followup: int = 6):
        super().__init__()
        self.hazard_fc      = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu           = nn.Softplus()

        # Upper-triangular mask: masked_hazards[b, i, j] = hazard[b,i] if j>=i else 0
        # Summing over dim=1 gives cumulative sum up to each year.
        mask = torch.tril(torch.ones(max_followup, max_followup), diagonal=0)
        mask = nn.Parameter(mask.t(), requires_grad=False)
        self.register_parameter("upper_triangular_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        hazards = self.relu(self.hazard_fc(x))          # [B, T]
        B, T    = hazards.size()
        expanded = hazards.unsqueeze(-1).expand(B, T, T) # [B, T, T]
        masked   = expanded * self.upper_triangular_mask  # [B, T, T]
        base     = self.base_hazard_fc(x)                 # [B, 1]
        cum_prob = masked.sum(dim=1) + base               # [B, T]
        return cum_prob
