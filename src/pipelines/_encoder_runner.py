from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from ..models.crf_encoder import CRFEncoder
from ..models.encoder import TokenClassificationEncoder

class EncoderRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        id2label: Dict[int, str],
        max_len: int = 256,
        overlap_ratio: float = 0.25,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.max_len = max_len
        self.overlap_ratio = overlap_ratio
        self.device = device
        self.is_crf = isinstance(model, CRFEncoder)

    def _encode_window(self, words: Sequence[str]):
        enc = self.tokenizer(
            list(words),
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        return enc, word_ids

    @torch.inference_mode()
    def _predict_window(self, words: Sequence[str]) -> List[str]:
        enc, word_ids = self._encode_window(words)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        word_tags: Dict[int, str] = {}

        if self.is_crf:
            wid_tensor = torch.tensor(
                [-1 if w is None else w for w in word_ids], dtype=torch.long
            ).unsqueeze(0).to(self.device)
            best_paths, word_indices = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                word_ids=wid_tensor,
            )
            for p_idx, wid in enumerate(word_indices[0]):
                if 0 <= wid < len(words):
                    word_tags[wid] = self.id2label[best_paths[0][p_idx]]
        else:
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=enc.get("token_type_ids"),
            )
            preds = logits.argmax(-1)[0].tolist()
            per_token = [self.id2label[p] for p in preds]
            prev_wid: int | None = None
            for t, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid != prev_wid:
                    word_tags[wid] = per_token[t]
                prev_wid = wid

        return [word_tags.get(i, "O") for i in range(len(words))]

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

    def predict_tokens(self, words: Sequence[str]) -> List[str]:
        if not words:
            return []

        enc, word_ids = self._encode_window(words)
        seen = set()
        for w in word_ids:
            if w is not None:
                seen.add(w)
        fits = len(seen) >= len(words)
        if fits:
            return self._predict_window(words)

        chunk_word_len = self._fit_chunk_size(words)
        overlap = max(1, int(chunk_word_len * self.overlap_ratio))
        step = max(1, chunk_word_len - overlap)

        votes: List[Dict[str, int]] = [{} for _ in range(len(words))]
        start = 0
        while start < len(words):
            end = min(start + chunk_word_len, len(words))
            chunk = words[start:end]
            chunk_tags = self._predict_window(chunk)
            for i, tag in enumerate(chunk_tags):
                global_i = start + i
                votes[global_i][tag] = votes[global_i].get(tag, 0) + 1
            if end == len(words):
                break
            start += step

        return [max(v.items(), key=lambda kv: kv[1])[0] if v else "O" for v in votes]
