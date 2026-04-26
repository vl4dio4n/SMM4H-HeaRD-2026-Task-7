from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, NamedTuple, Sequence

from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

class Entity(NamedTuple):
    e_type: str
    start: int
    end: int

def bio_to_entities(tags: Sequence[str]) -> List[Entity]:
    entities: List[Entity] = []
    start: int | None = None
    cur: str | None = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if cur is not None:
                entities.append(Entity(cur, start, i - 1))
            cur = tag[2:]
            start = i
        elif tag.startswith("I-"):
            if cur == tag[2:]:
                continue
            if cur is not None:
                entities.append(Entity(cur, start, i - 1))
            cur = tag[2:]
            start = i
        else:
            if cur is not None:
                entities.append(Entity(cur, start, i - 1))
                cur = None
                start = None

    if cur is not None:
        entities.append(Entity(cur, start, len(tags) - 1))
    return entities

def compute_strict_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Dict[str, float]:
    return {
        "precision_strict": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_strict": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_strict": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

def _overlap(a: Entity, b: Entity) -> int:
    if a.e_type != b.e_type:
        return 0
    return max(0, min(a.end, b.end) - max(a.start, b.start) + 1)

def compute_relaxed_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tp_overlap": 0, "total_true": 0, "total_pred": 0}
    )

    for gold, pred in zip(y_true, y_pred):
        true_ents = bio_to_entities(gold)
        pred_ents = bio_to_entities(pred)
        matched: set[int] = set()

        for t in true_ents:
            for j, p in enumerate(pred_ents):
                if j in matched:
                    continue
                ov = _overlap(t, p)
                if ov > 0:
                    agg[t.e_type]["tp_overlap"] += ov
                    matched.add(j)
            agg[t.e_type]["total_true"] += (t.end - t.start + 1)

        for p in pred_ents:
            agg[p.e_type]["total_pred"] += (p.end - p.start + 1)

    results: Dict[str, Dict[str, float]] = {}
    sum_tp = sum_true = sum_pred = 0
    for etype, v in agg.items():
        tp, tt, tp_pred = v["tp_overlap"], v["total_true"], v["total_pred"]
        precision = tp / tp_pred if tp_pred else 0.0
        recall = tp / tt if tt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[etype] = {"precision": precision, "recall": recall, "f1": f1}
        sum_tp += tp
        sum_true += tt
        sum_pred += tp_pred

    overall_p = sum_tp / sum_pred if sum_pred else 0.0
    overall_r = sum_tp / sum_true if sum_true else 0.0
    overall_f1 = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if (overall_p + overall_r)
        else 0.0
    )
    results["Overall"] = {"precision": overall_p, "recall": overall_r, "f1": overall_f1}
    return results

def evaluate_predictions(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    print_report: bool = False,
) -> Dict[str, object]:
    strict = compute_strict_metrics(y_true, y_pred)
    relaxed = compute_relaxed_metrics(y_true, y_pred)
    if print_report:
        print("=== STRICT (seqeval) ===")
        for k, v in strict.items():
            print(f"  {k}: {v:.4f}")
        print("\n=== Per-entity classification report ===")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
        print("=== RELAXED (token overlap) ===")
        for etype, m in relaxed.items():
            print(f"  {etype}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    return {"strict": strict, "relaxed": relaxed}
