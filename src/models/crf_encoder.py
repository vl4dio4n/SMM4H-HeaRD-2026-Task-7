from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchcrf import CRF

from .encoder import TokenClassificationEncoder

def _first_subword_positions(
    word_ids: torch.Tensor,
) -> List[List[int]]:
    B, T = word_ids.shape
    out: List[List[int]] = []
    for b in range(B):
        seen = set()
        positions: List[int] = []
        for t in range(T):
            wid = int(word_ids[b, t].item())
            if wid < 0:
                continue
            if wid not in seen:
                seen.add(wid)
                positions.append(t)
        out.append(positions)
    return out

class CRFEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = TokenClassificationEncoder(
            model_name=model_name, num_labels=num_labels, dropout=dropout
        )
        self.num_labels = num_labels
        self.crf = CRF(num_labels, batch_first=True)

    @staticmethod
    def _labels_to_first_subword(
        labels: torch.Tensor,
        word_ids: torch.Tensor,
    ) -> torch.Tensor:
        return labels

    def _pack(
        self,
        emissions: torch.Tensor,
        word_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        first_positions = _first_subword_positions(word_ids)
        max_words = max((len(p) for p in first_positions), default=1)
        B, T, C = emissions.shape
        device = emissions.device

        packed_em = torch.zeros(B, max_words, C, device=device, dtype=emissions.dtype)
        packed_mask = torch.zeros(B, max_words, dtype=torch.bool, device=device)
        packed_labels = (
            torch.zeros(B, max_words, dtype=torch.long, device=device)
            if labels is not None
            else None
        )
        word_indices: List[List[int]] = []

        for b, positions in enumerate(first_positions):
            if not positions:
                packed_mask[b, 0] = True
                word_indices.append([])
                continue
            for w, t in enumerate(positions):
                packed_em[b, w] = emissions[b, t]
                if packed_labels is not None:
                    lab = int(labels[b, t].item())
                    packed_labels[b, w] = 0 if lab == -100 else lab
            packed_mask[b, : len(positions)] = True
            wids_for_item = [int(word_ids[b, t].item()) for t in positions]
            word_indices.append(wids_for_item)

        return packed_em, packed_mask, packed_labels, word_indices

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
    ):
        emissions = self.backbone(input_ids, attention_mask, token_type_ids)

        if word_ids is None:
            raise ValueError("CRFEncoder.forward requires `word_ids`.")

        packed_em, packed_mask, packed_labels, word_indices = self._pack(
            emissions, word_ids, labels
        )

        if labels is not None:
            nll = -self.crf(packed_em, packed_labels, mask=packed_mask, reduction="mean")
            return nll, word_indices

        best_paths: List[List[int]] = self.crf.decode(packed_em, mask=packed_mask)
        return best_paths, word_indices
