from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml
from transformers import AutoTokenizer

from src.data.processor import (
    LABELS_3,
    LABELS_5,
    collapse_5_to_3,
    label_to_id,
    load_split,
)
from src.classifiers.svm_span import SVMSpanClassifier
from src.evaluation.metrics import evaluate_predictions
from src.llm.factory import build_llm
from src.llm.few_shot import build_few_shot_pool
from src.models.factory import build_model
from src.pipelines.audit import AuditPipeline
from src.pipelines.direct import DirectNERPipeline
from src.pipelines.svm_audit import SVMAuditPipeline
from src.pipelines.svm_two_stage import SVMTwoStagePipeline
from src.pipelines.two_stage import TwoStagePipeline

def load_experiment(exp_dir: Path) -> Dict:
    with open(exp_dir / "config.yaml") as f:
        return yaml.safe_load(f)

def build_pipeline(cfg: Dict, exp_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["model_name"])
    model = build_model(cfg["encoder"])

    ckpt = exp_dir / "checkpoints" / "best" / "model.pt"
    if not ckpt.exists():
        ckpt = exp_dir / "checkpoints" / "last" / "model.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    num_labels = int(cfg["encoder"]["num_labels"])
    labels = LABELS_5 if num_labels == 5 else LABELS_3
    l2id = label_to_id(labels)
    id2label = {i: lab for lab, i in l2id.items()}
    max_len = int(cfg["encoder"].get("max_len", 256))

    pipeline_kind = cfg.get("pipeline", "direct")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if pipeline_kind == "direct":
        return DirectNERPipeline(
            model=model, tokenizer=tokenizer, id2label=id2label,
            max_len=max_len, device=device,
        ), id2label

    if pipeline_kind in {"svm_two_stage", "svm_audit"}:
        svm_cfg = cfg.get("svm", {})
        svm_path = svm_cfg.get("model_path") or str(exp_dir / "svm.joblib")
        svm = SVMSpanClassifier.load(svm_path)
        if pipeline_kind == "svm_two_stage":
            pipeline = SVMTwoStagePipeline(
                model=model, tokenizer=tokenizer, id2label_3class=id2label,
                svm=svm, max_len=max_len, device=device,
            )
        else:
            pipeline = SVMAuditPipeline(
                model=model, tokenizer=tokenizer, id2label_5class=id2label,
                svm=svm, max_len=max_len, device=device,
            )
        return pipeline, id2label

    llm = build_llm(cfg.get("llm", {}))

    few_shot_cfg = cfg.get("few_shot") or {}
    few_shot_examples = None
    k_per_class = int(few_shot_cfg.get("k_per_class", 0))
    if k_per_class > 0:
        train_path = (
            cfg.get("data", {}).get("train_path", "datasets/new_train_data.csv")
        )
        few_shot_examples = build_few_shot_pool(
            train_path=train_path,
            k_per_class=k_per_class,
            include_neither=bool(few_shot_cfg.get("include_neither", True)),
            seed=int(few_shot_cfg.get("seed", 42)),
            max_post_chars=int(few_shot_cfg.get("max_post_chars", 600)),
            max_span_chars=int(few_shot_cfg.get("max_span_chars", 120)),
        )
        print(
            f"Few-shot: loaded {len(few_shot_examples)} examples "
            f"(k_per_class={k_per_class}, include_neither="
            f"{few_shot_cfg.get('include_neither', True)})"
        )

    if pipeline_kind == "two_stage":
        pipeline = TwoStagePipeline(
            model=model, tokenizer=tokenizer, id2label_3class=id2label,
            llm=llm, max_len=max_len, device=device,
            few_shot_examples=few_shot_examples,
        )
    elif pipeline_kind == "audit":
        pipeline = AuditPipeline(
            model=model, tokenizer=tokenizer, id2label_5class=id2label,
            llm=llm, max_len=max_len, device=device,
            few_shot_examples=few_shot_examples,
        )
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_kind}")
    return pipeline, id2label

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, type=str)
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--save-predictions", type=str, default=None,
                    help="Optional CSV path to dump (ID, gold_ner_tags, predicted_ner_tags)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    cfg = load_experiment(exp_dir)

    split_path = (
        cfg.get("data", {}).get(f"{args.split}_path")
        or f"datasets/new_{args.split}_data.csv"
    )
    df = load_split(split_path)

    pipeline_kind = cfg.get("pipeline", "direct")
    num_labels = int(cfg["encoder"]["num_labels"])
    collapse_gold = pipeline_kind == "direct" and num_labels == 3
    gold_tags = [
        collapse_5_to_3(list(t)) if collapse_gold else list(t)
        for t in df["ner_tags"]
    ]

    pipeline, _ = build_pipeline(cfg, exp_dir)
    print(f"Predicting {len(df)} segments from {split_path} ...")
    preds = [pipeline.predict_tokens(list(tokens)) for tokens in df["tokens"]]

    metrics = evaluate_predictions(gold_tags, preds, print_report=True)
    out_metrics = exp_dir / f"eval_{args.split}.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {out_metrics}")

    if args.save_predictions:
        import pandas as pd
        pd.DataFrame({
            "ID": df["ID"],
            "gold_ner_tags": gold_tags,
            "predicted_ner_tags": preds,
        }).to_csv(args.save_predictions, index=False)
        print(f"Predictions written to {args.save_predictions}")

if __name__ == "__main__":
    main()
