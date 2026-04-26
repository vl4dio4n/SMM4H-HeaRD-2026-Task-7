from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from transformers import pipeline

from .base import CLASSES, BaseLLMClassifier, FewShotExample

_HYPOTHESES = {
    "ClinicalImpacts": "This text describes a physical or medical impact of substance use "
                       "(e.g., withdrawal, overdose, addiction, treatment).",
    "SocialImpacts":   "This text describes a social impact of substance use "
                       "(e.g., job loss, homelessness, broken relationship, legal trouble).",
    "Neither":         "This text is not describing any impact of substance use.",
}

class NLIClassifier(BaseLLMClassifier):
    def __init__(self, model_name: str = "facebook/bart-large-mnli") -> None:
        self.pipe = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )
        self.labels = list(CLASSES)

    def classify_span(
        self,
        post: str,
        span: str,
        current_label: Optional[str] = None,
        examples: Optional[List[FewShotExample]] = None,
    ) -> str:
        text = f"{post}\n\nSpan of interest: {span}"
        result = self.pipe(text, candidate_labels=self.labels, multi_label=False)
        return result["labels"][0]

    def classify_batch(
        self,
        inputs: List[Tuple[str, str, Optional[str]]],
        examples: Optional[List[FewShotExample]] = None,
    ) -> List[str]:
        texts = [f"{p}\n\nSpan of interest: {s}" for p, s, _ in inputs]
        results = self.pipe(texts, candidate_labels=self.labels, multi_label=False)
        if isinstance(results, dict):
            results = [results]
        return [r["labels"][0] for r in results]
