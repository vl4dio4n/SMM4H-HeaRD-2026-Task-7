import pandas as pd
import ast
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from typing import List, NamedTuple, Dict
from collections import defaultdict

def _to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        return ast.literal_eval(x)
    raise TypeError(f"Unsupported cell type: {type(x)}")

def evaluate_test_strict_ner(
    df: pd.DataFrame,
    gold_col: str = "test",
    pred_col: str = "prediction",
    print_report: bool = True,
):
    y_true = df[gold_col].apply(_to_list).tolist()
    y_pred = df[pred_col].apply(_to_list).tolist()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Row mismatch: gold rows={len(y_true)} pred rows={len(y_pred)}")

    for i, (g, p) in enumerate(zip(y_true, y_pred)):
        if len(g) != len(p):
            raise ValueError(
                f"Token length mismatch at row {i}: gold={len(g)} pred={len(p)}\n"
                f"gold: {g}\n"
                f"pred: {p}"
            )

    metrics = {
        "precision_strict": precision_score(y_true, y_pred),
        "recall_strict": recall_score(y_true, y_pred),
        "f1_strict": f1_score(y_true, y_pred),
        "accuracy_strict": accuracy_score(y_true, y_pred),
    }

    if print_report:
        print("=== STRICT ENTITY NER METRICS (seqeval) ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("\n=== PER-LABEL REPORT ===")
        print(classification_report(y_true, y_pred, digits=4))

    return metrics

class Entity(NamedTuple):
    e_type: str
    start_offset: int
    end_offset: int

def bio_to_entities(bio_tags: List[str]) -> List[Entity]:
    entities = []
    start = None
    entity_type = None

    for i, tag in enumerate(bio_tags):
        if tag.startswith("B-"):
            if entity_type is not None:
                entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=i - 1))
            entity_type = tag[2:]
            start = i
        elif tag.startswith("I-") and entity_type == tag[2:]:

            continue
        elif tag.startswith("I-") and entity_type != tag[2:]:

            if entity_type is not None:
                entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=i - 1))
            entity_type = tag[2:]
            start = i
        elif tag == "O":
            if entity_type is not None:
                entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=i - 1))
                entity_type = None
                start = None

    if entity_type is not None:
        entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=len(bio_tags) - 1))

    return entities

def relaxed_overlap(entity1: Entity, entity2: Entity) -> float:
    if entity1.e_type != entity2.e_type:
        return 0

    return max(0, min(entity1.end_offset, entity2.end_offset) - max(entity1.start_offset, entity2.start_offset) + 1)

def calculate_f1_per_entity_covering_all(gold_labels: List[List[str]], pred_labels: List[List[str]]) -> dict:
    aggregated_results = defaultdict(lambda: {"TP_overlap": 0, "Total_True_Length": 0, "Total_Pred_Length": 0})

    for gold, pred in zip(gold_labels, pred_labels):

        true_entities = bio_to_entities(gold)
        pred_entities = bio_to_entities(pred)

        matched_pred_indices = set()
        for true_entity in true_entities:

            for i, pred_entity in enumerate(pred_entities):

                if i in matched_pred_indices:
                    continue
                overlap = relaxed_overlap(true_entity, pred_entity)
                if overlap > 0:
                    aggregated_results[true_entity.e_type]["TP_overlap"] += overlap
                    matched_pred_indices.add(i)
            aggregated_results[true_entity.e_type]["Total_True_Length"] += (true_entity.end_offset - true_entity.start_offset + 1)

        for pred_entity in pred_entities:
            aggregated_results[pred_entity.e_type]["Total_Pred_Length"] += (pred_entity.end_offset - pred_entity.start_offset + 1)

    final_results = {}
    overall_tp_overlap = 0
    overall_true_length = 0
    overall_pred_length = 0

    for entity_type, values in aggregated_results.items():
        precision = values["TP_overlap"] / values["Total_Pred_Length"] if values["Total_Pred_Length"] > 0 else 0.0
        recall = values["TP_overlap"] / values["Total_True_Length"] if values["Total_True_Length"] > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        overall_tp_overlap += values["TP_overlap"]
        overall_true_length += values["Total_True_Length"]
        overall_pred_length += values["Total_Pred_Length"]

        final_results[entity_type] = {
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-Score": round(f1, 3),
            "Coverage": f"{values['TP_overlap']}/{values['Total_True_Length']}"
        }

    overall_precision = overall_tp_overlap / overall_pred_length if overall_pred_length > 0 else 0.0
    overall_recall = overall_tp_overlap / overall_true_length if overall_true_length > 0 else 0.0
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    final_results["Overall"] = {
        "Precision": round(overall_precision, 3),
        "Recall": round(overall_recall, 3),
        "F1-Score": round(overall_f1, 3),
        "Coverage": f"{overall_tp_overlap}/{overall_true_length}"
    }
    return final_results

if __name__ == "__main__":
    print("Loading datasets...")

    dev_df = pd.read_csv("datasets/new_dev_data.csv")

    preds_df = pd.read_csv("submission.csv")

    df = pd.merge(dev_df, preds_df, on="ID")

    gold_col_name = 'ner_tags'
    pred_col_name = 'predicted_ner_tags'

    print(f"Evaluating {len(df)} sequences...\n")

    metrics = evaluate_test_strict_ner(
        df,
        gold_col=gold_col_name,
        pred_col=pred_col_name,
        print_report=False
    )
    micro_f1 = metrics['f1_strict']

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    gold_tags_list = df[gold_col_name].apply(_to_list).tolist()
    prediction_list = df[pred_col_name].apply(_to_list).tolist()

    results_per_entity = calculate_f1_per_entity_covering_all(gold_tags_list, prediction_list)

    print("\nF1 Score Results Per Entity (Relaxed Overlap):")
    for entity, entity_metrics in results_per_entity.items():
        if entity != "Overall":
            print(f"Entity Type: {entity}")
            for metric, value in entity_metrics.items():
                print(f"  {metric}: {value}")
            print()

    overall_relax_f1 = results_per_entity['Overall']['F1-Score']

    print("=" * 50)
    print(f"Micro F1 (Strict):     {micro_f1:.4f}")
    print(f"Overall Relax F1:      {overall_relax_f1:.4f}")
    print("=" * 50)

