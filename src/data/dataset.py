from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

class SMM4HDataset(Dataset):

    def __init__(
        self,
        tokens_list: Sequence[Sequence[str]],
        tags_list: Optional[Sequence[Sequence[str]]],
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        max_len: int = 256,
        ids: Optional[Sequence[str]] = None,
        label_all_subwords: bool = False,
    ) -> None:
        assert tags_list is None or len(tokens_list) == len(tags_list), \
            "tokens and tags lengths must match"
        self.tokens_list = [list(t) for t in tokens_list]
        self.tags_list = [list(t) for t in tags_list] if tags_list is not None else None
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.ids = list(ids) if ids is not None else None
        self.label_all_subwords = label_all_subwords

    def __len__(self) -> int:
        return len(self.tokens_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        words = self.tokens_list[idx]
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        word_ids = encoding.word_ids(batch_index=0)

        if self.tags_list is not None:
            word_tags = self.tags_list[idx]
            label_ids: List[int] = []
            prev_word_id: Optional[int] = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev_word_id:
                    label_ids.append(self.label2id[word_tags[wid]])
                else:
                    if self.label_all_subwords:
                        base = word_tags[wid]
                        if base.startswith("B-"):
                            base = "I-" + base[2:]
                        label_ids.append(self.label2id.get(base, -100))
                    else:
                        label_ids.append(-100)
                prev_word_id = wid
            item["labels"] = torch.tensor(label_ids, dtype=torch.long)

        item["word_ids"] = torch.tensor(
            [-1 if w is None else w for w in word_ids], dtype=torch.long
        )
        item["num_words"] = torch.tensor(len(words), dtype=torch.long)
        if self.ids is not None:
            item["segment_id"] = self.ids[idx]
        return item

def build_collate_fn(tokenizer: PreTrainedTokenizerBase):
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        max_len = max(b["input_ids"].size(0) for b in batch)
        out: Dict[str, torch.Tensor | list] = {}
        keys_token = ["input_ids", "attention_mask", "token_type_ids", "labels", "word_ids"]
        for key in keys_token:
            if key not in batch[0]:
                continue
            if key == "input_ids":
                fill = pad_id if pad_id is not None else 0
            elif key == "labels":
                fill = -100
            elif key == "word_ids":
                fill = -1
            else:
                fill = 0
            padded = torch.full(
                (len(batch), max_len), fill, dtype=batch[0][key].dtype
            )
            for i, b in enumerate(batch):
                t = b[key]
                padded[i, : t.size(0)] = t
            out[key] = padded

        out["num_words"] = torch.stack([b["num_words"] for b in batch])
        if "segment_id" in batch[0]:
            out["segment_id"] = [b["segment_id"] for b in batch]
        return out

    return collate
