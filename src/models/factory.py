from __future__ import annotations

from typing import Dict

import torch.nn as nn

from .crf_encoder import CRFEncoder
from .encoder import TokenClassificationEncoder

def build_model(encoder_cfg: Dict) -> nn.Module:
    model_name = encoder_cfg["model_name"]
    num_labels = int(encoder_cfg["num_labels"])
    dropout = float(encoder_cfg.get("dropout", 0.1))
    if encoder_cfg.get("use_crf", False):
        return CRFEncoder(model_name=model_name, num_labels=num_labels, dropout=dropout)
    return TokenClassificationEncoder(
        model_name=model_name, num_labels=num_labels, dropout=dropout
    )
