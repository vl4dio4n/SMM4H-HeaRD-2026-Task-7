from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from ..models.crf_encoder import CRFEncoder
from ..models.encoder import TokenClassificationEncoder

Span = Tuple[int, int]

class SpanFeatureExtractor:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = 256,
        overlap_ratio: float = 0.25,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap_ratio = overlap_ratio
        self.device = device
        self.backbone = self._resolve_backbone(model)

    @staticmethod
    def _resolve_backbone(model: torch.nn.Module) -> TokenClassificationEncoder:
        if isinstance(model, CRFEncoder):
            return model.backbone
        if isinstance(model, TokenClassificationEncoder):
            return model
        raise TypeError(f"Unsupported model type: {type(model)}")

    @torch.inference_mode()
    def _run_window(
        self, words: Sequence[str],
    ) -> Tuple[torch.Tensor, List[Optional[int]]]:
        enc = self.tokenizer(
            list(words),
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        enc_on_device = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.backbone.encode(
            input_ids=enc_on_device["input_ids"],
            attention_mask=enc_on_device["attention_mask"],
            token_type_ids=enc_on_device.get("token_type_ids"),
        )
        return hidden[0].float().cpu(), word_ids

    def _fit_chunk_size(self, words: Sequence[str]) -> int:
        lo, hi = 1, len(words)
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            enc = self.tokenizer(
                list(words[:mid]),
                is_split_into_words=True,
                truncation=False,
                add_special_tokens=True,
            )
            if len(enc["input_ids"]) <= self.max_len:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return max(1, best)

    def word_embeddings(self, words: Sequence[str]) -> np.ndarray:
        if not words:
            hidden = self.backbone.config.hidden_size
            return np.zeros((0, hidden), dtype=np.float32)

        enc, word_ids = self._run_window(words)
        seen = {w for w in word_ids if w is not None}
        fits_in_single_window = len(seen) >= len(words)

        if fits_in_single_window:
            return self._pool_single_window(enc, word_ids, len(words))

        chunk = self._fit_chunk_size(words)
        overlap = max(1, int(chunk * self.overlap_ratio))
        step = max(1, chunk - overlap)
        H = self.backbone.config.hidden_size
        accum = np.zeros((len(words), H), dtype=np.float32)
        counts = np.zeros(len(words), dtype=np.float32)

        start = 0
        while start < len(words):
            end = min(start + chunk, len(words))
            window_words = words[start:end]
            window_hidden, window_wids = self._run_window(window_words)
            window_pool = self._pool_single_window(
                window_hidden, window_wids, len(window_words)
            )
            for local_w, global_w in enumerate(range(start, end)):
                if not np.any(window_pool[local_w]):
                    continue
                accum[global_w] += window_pool[local_w]
                counts[global_w] += 1
            if end == len(words):
                break
            start += step

        counts = np.where(counts == 0, 1.0, counts)
        return (accum.T / counts).T

    @staticmethod
    def _pool_single_window(
        hidden: torch.Tensor,
        word_ids: List[Optional[int]],
        n_words: int,
    ) -> np.ndarray:
        H = hidden.shape[-1]
        sums = np.zeros((n_words, H), dtype=np.float32)
        counts = np.zeros(n_words, dtype=np.float32)
        hidden_np = hidden.numpy()
        for t, wid in enumerate(word_ids):
            if wid is None or wid >= n_words:
                continue
            sums[wid] += hidden_np[t]
            counts[wid] += 1.0
        counts = np.where(counts == 0, 1.0, counts)
        return (sums.T / counts).T

    def extract(self, words: Sequence[str], spans: Sequence[Span]) -> np.ndarray:
        if not spans:
            hidden = self.backbone.config.hidden_size
            return np.zeros((0, hidden), dtype=np.float32)

        word_embs = self.word_embeddings(words)
        out = np.zeros((len(spans), word_embs.shape[-1]), dtype=np.float32)
        for i, (s, e) in enumerate(spans):
            s = max(0, s)
            e = min(len(words) - 1, e)
            if s > e:
                continue
            out[i] = word_embs[s:e + 1].mean(axis=0)
        return out
