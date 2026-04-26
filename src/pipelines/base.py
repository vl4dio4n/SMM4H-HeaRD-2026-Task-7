from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

class BasePipeline(ABC):
    @abstractmethod
    def predict_tokens(self, tokens: Sequence[str]) -> List[str]:
        pass

    def predict_batch(self, batch_tokens: List[Sequence[str]]) -> List[List[str]]:
        return [self.predict_tokens(t) for t in batch_tokens]
