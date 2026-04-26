from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from ..data.processor import load_split
from .base import CLASSES, FewShotExample

POSITIVE_CLASSES: Tuple[str, ...] = ("ClinicalImpacts", "SocialImpacts")

def _extract_spans(tags: Sequence[str]) -> List[Tuple[int, int, str]]:
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

def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

def _is_content_token(tok: str) -> bool:
    return any(c.isalpha() for c in tok) and len(tok) >= 3

def _pick_neither_span(
    tokens: Sequence[str], rng: random.Random, min_len: int = 2, max_len: int = 4,
) -> Optional[Tuple[int, int]]:
    content_idx = [i for i, t in enumerate(tokens) if _is_content_token(t)]
    if len(content_idx) < min_len:
        return None
    span_len = rng.randint(min_len, min(max_len, len(content_idx)))
    start_content = rng.randint(0, len(content_idx) - span_len)
    start, end = content_idx[start_content], content_idx[start_content + span_len - 1]
    return start, end

def build_few_shot_pool(
    train_path: str | Path,
    k_per_class: int,
    include_neither: bool,
    seed: int = 42,
    max_post_chars: int = 600,
    max_span_chars: int = 120,
) -> List[FewShotExample]:
    df = load_split(train_path)
    rng = random.Random(seed)

    per_class: dict[str, List[FewShotExample]] = {c: [] for c in POSITIVE_CLASSES}
    if include_neither:
        per_class["Neither"] = []

    pos_candidates: dict[str, List[Tuple[int, int, int]]] = {c: [] for c in POSITIVE_CLASSES}
    neither_candidates: List[int] = []

    for row_idx, row in df.iterrows():
        tokens = list(row["tokens"])
        tags = list(row["ner_tags"])
        spans = _extract_spans(tags)
        if spans:
            for s, e, cls in spans:
                if cls in POSITIVE_CLASSES:
                    pos_candidates[cls].append((int(row_idx), s, e))
        else:
            if len(tokens) >= 4:
                neither_candidates.append(int(row_idx))

    examples: List[FewShotExample] = []
    for cls in POSITIVE_CLASSES:
        pool = pos_candidates[cls]
        rng.shuffle(pool)
        for row_idx, s, e in pool[:k_per_class]:
            tokens = list(df.iloc[row_idx]["tokens"])
            post = _truncate(" ".join(tokens), max_post_chars)
            span_text = _truncate(" ".join(tokens[s:e + 1]), max_span_chars)
            examples.append(FewShotExample(post=post, span=span_text, label=cls))

    if include_neither:
        rng.shuffle(neither_candidates)
        added = 0
        for row_idx in neither_candidates:
            if added >= k_per_class:
                break
            tokens = list(df.iloc[row_idx]["tokens"])
            span_range = _pick_neither_span(tokens, rng)
            if span_range is None:
                continue
            s, e = span_range
            post = _truncate(" ".join(tokens), max_post_chars)
            span_text = _truncate(" ".join(tokens[s:e + 1]), max_span_chars)
            examples.append(FewShotExample(post=post, span=span_text, label="Neither"))
            added += 1

    rng.shuffle(examples)
    return examples
