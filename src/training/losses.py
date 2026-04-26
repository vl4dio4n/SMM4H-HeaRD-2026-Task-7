from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCE(nn.Module):
    def __init__(self, weights: Optional[Sequence[float]] = None, ignore_index: int = -100):
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
        self.ignore_index = ignore_index
        self.register_buffer("weights", w if w is not None else torch.empty(0), persistent=False)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w = self.weights.to(logits.device) if self.weights.numel() > 0 else None
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            weight=w,
            ignore_index=self.ignore_index,
        )

class FocalLoss(nn.Module):

    def __init__(
        self,
        gamma: float = 2.0,
        weights: Optional[Sequence[float]] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        w = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
        self.register_buffer("weights", w if w is not None else torch.empty(0), persistent=False)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)
        mask = flat_labels != self.ignore_index
        if mask.sum() == 0:
            return flat_logits.sum() * 0.0
        flat_logits = flat_logits[mask]
        flat_labels = flat_labels[mask]

        log_probs = F.log_softmax(flat_logits, dim=-1)
        log_pt = log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal = ((1 - pt) ** self.gamma) * (-log_pt)

        if self.weights.numel() > 0:
            alpha = self.weights.to(focal.device)[flat_labels]
            focal = alpha * focal

        return focal.mean()

def build_loss(cfg: dict) -> nn.Module:
    kind = cfg.get("type", "ce")
    if kind == "ce":
        return WeightedCE(weights=None)
    if kind == "weighted_ce":
        return WeightedCE(weights=cfg["weights"])
    if kind == "focal":
        return FocalLoss(gamma=cfg.get("gamma", 2.0), weights=cfg.get("weights"))
    raise ValueError(f"Unknown loss type: {kind}")
