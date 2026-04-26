from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class TokenClassificationEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)

        self.backbone = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self.config.model_type not in {"roberta", "deberta-v2"}:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.backbone(**kwargs).last_hidden_state
        return self.dropout(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.encode(input_ids, attention_mask, token_type_ids)
        return self.classifier(hidden)
