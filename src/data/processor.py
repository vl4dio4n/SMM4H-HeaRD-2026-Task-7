from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

LABELS_5: Tuple[str, ...] = (
    "O",
    "B-ClinicalImpacts",
    "I-ClinicalImpacts",
    "B-SocialImpacts",
    "I-SocialImpacts",
)

LABELS_3: Tuple[str, ...] = (
    "O",
    "B-Impact",
    "I-Impact",
)

def label_to_id(labels: Tuple[str, ...]) -> Dict[str, int]:
    return {lab: i for i, lab in enumerate(labels)}

def id_to_label(labels: Tuple[str, ...]) -> Dict[int, str]:
    return {i: lab for i, lab in enumerate(labels)}

def collapse_5_to_3(tags: List[str]) -> List[str]:
    out: List[str] = []
    for t in tags:
        if t == "O":
            out.append("O")
        elif t.startswith("B-"):
            out.append("B-Impact")
        elif t.startswith("I-"):
            out.append("I-Impact")
        else:
            out.append("O")
    return out

def _to_list(cell):
    if isinstance(cell, list):
        return cell
    return ast.literal_eval(cell)

def load_split(path: str | Path, has_labels: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tokens"] = df["tokens"].apply(_to_list)
    if has_labels:
        if "ner_tags" in df.columns:
            df["ner_tags"] = df["ner_tags"].apply(_to_list)
        if "labels" in df.columns:
            df["labels"] = df["labels"].apply(_to_list)
    return df.reset_index(drop=True)
