from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from ._encoder_runner import EncoderRunner
from .base import BasePipeline

class DirectNERPipeline(BasePipeline):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        id2label: Dict[int, str],
        max_len: int = 256,
        overlap_ratio: float = 0.25,
        device: str = "cuda",
    ) -> None:
        self.runner = EncoderRunner(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            max_len=max_len,
            overlap_ratio=overlap_ratio,
            device=device,
        )

    def predict_tokens(self, tokens: Sequence[str]) -> List[str]:
        return self.runner.predict_tokens(tokens)
