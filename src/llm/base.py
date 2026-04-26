from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

CLASSES: Tuple[str, str, str] = ("ClinicalImpacts", "SocialImpacts", "Neither")

@dataclass
class FewShotExample:
    post: str
    span: str
    label: str

class BaseLLMClassifier(ABC):

    @abstractmethod
    def classify_span(
        self,
        post: str,
        span: str,
        current_label: Optional[str] = None,
        examples: Optional[List[FewShotExample]] = None,
    ) -> str:
        pass

    def classify_batch(
        self,
        inputs: List[Tuple[str, str, Optional[str]]],
        examples: Optional[List[FewShotExample]] = None,
    ) -> List[str]:
        return [
            self.classify_span(post, span, current_label=cur, examples=examples)
            for post, span, cur in inputs
        ]

    def close(self) -> None:
        pass
