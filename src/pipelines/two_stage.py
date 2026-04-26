from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from ..llm.base import BaseLLMClassifier, FewShotExample
from ._encoder_runner import EncoderRunner
from .base import BasePipeline

def _extract_impact_spans(tags: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, tag in enumerate(tags):
        if tag == "B-Impact":
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == "I-Impact" and start is not None:
            continue
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(tags) - 1))
    return spans

class TwoStagePipeline(BasePipeline):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        id2label_3class: Dict[int, str],
        llm: BaseLLMClassifier,
        max_len: int = 256,
        overlap_ratio: float = 0.25,
        device: str = "cuda",
        few_shot_examples: Optional[List[FewShotExample]] = None,
    ) -> None:
        self.runner = EncoderRunner(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label_3class,
            max_len=max_len,
            overlap_ratio=overlap_ratio,
            device=device,
        )
        self.llm = llm
        self.examples = few_shot_examples

    def predict_tokens(self, tokens: Sequence[str]) -> List[str]:
        boundary_tags = self.runner.predict_tokens(tokens)
        spans = _extract_impact_spans(boundary_tags)
        if not spans:
            return ["O"] * len(tokens)

        post = " ".join(tokens)
        inputs = [
            (post, " ".join(tokens[s:e + 1]), None) for s, e in spans
        ]
        labels = self.llm.classify_batch(inputs, examples=self.examples)

        out: List[str] = ["O"] * len(tokens)
        for (s, e), lab in zip(spans, labels):
            if lab == "Neither":
                continue
            out[s] = f"B-{lab}"
            for k in range(s + 1, e + 1):
                out[k] = f"I-{lab}"
        return out
