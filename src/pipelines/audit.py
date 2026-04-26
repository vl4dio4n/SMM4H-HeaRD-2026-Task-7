from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from ..llm.base import BaseLLMClassifier, CLASSES, FewShotExample
from ._encoder_runner import EncoderRunner
from .base import BasePipeline

def _extract_typed_spans(tags: Sequence[str]) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    start: Optional[int] = None
    cur_type: Optional[str] = None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None and cur_type is not None:
                spans.append((start, i - 1, cur_type))
            cur_type = tag[2:]
            start = i
        elif tag.startswith("I-") and cur_type is not None and tag[2:] == cur_type:
            continue
        else:
            if start is not None and cur_type is not None:
                spans.append((start, i - 1, cur_type))
            start = None
            cur_type = None
    if start is not None and cur_type is not None:
        spans.append((start, len(tags) - 1, cur_type))
    return spans

class AuditPipeline(BasePipeline):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        id2label_5class: Dict[int, str],
        llm: BaseLLMClassifier,
        max_len: int = 256,
        overlap_ratio: float = 0.25,
        device: str = "cuda",
        few_shot_examples: Optional[List[FewShotExample]] = None,
    ) -> None:
        self.runner = EncoderRunner(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label_5class,
            max_len=max_len,
            overlap_ratio=overlap_ratio,
            device=device,
        )
        self.llm = llm
        self.examples = few_shot_examples

    def predict_tokens(self, tokens: Sequence[str]) -> List[str]:
        raw_tags = self.runner.predict_tokens(tokens)
        spans = _extract_typed_spans(raw_tags)
        if not spans:
            return raw_tags

        post = " ".join(tokens)
        inputs = [
            (post, " ".join(tokens[s:e + 1]), cur) for s, e, cur in spans
        ]
        llm_out = self.llm.classify_batch(inputs, examples=self.examples)

        out = list(raw_tags)
        for (s, e, cur), decision in zip(spans, llm_out):
            if decision == "Neither":
                for k in range(s, e + 1):
                    out[k] = "O"
            elif decision in CLASSES and decision != cur:
                out[s] = f"B-{decision}"
                for k in range(s + 1, e + 1):
                    out[k] = f"I-{decision}"
        return out
