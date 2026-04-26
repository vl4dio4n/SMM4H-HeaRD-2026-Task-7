from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

from src.data.processor import LABELS_3, LABELS_5, label_to_id, load_split
from src.evaluation.metrics import evaluate_predictions
from src.models.factory import build_model
from src.pipelines.direct import DirectNERPipeline

DEFAULT_ID_PATTERN = r"^test_\d{1,3}$"

def _load_member(parent: Path, seed: int, cfg: Dict) -> DirectNERPipeline:
    seed_dir = parent / f"seed_{seed}"
    ckpt = seed_dir / "checkpoints" / "best" / "model.pt"
    if not ckpt.exists():
        ckpt = seed_dir / "checkpoints" / "last" / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint for seed {seed} under {seed_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["model_name"])
    model = build_model(cfg["encoder"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    num_labels = int(cfg["encoder"]["num_labels"])
    labels = LABELS_5 if num_labels == 5 else LABELS_3
    id2label = {i: lab for lab, i in label_to_id(labels).items()}
    max_len = int(cfg["encoder"].get("max_len", 256))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return DirectNERPipeline(
        model=model, tokenizer=tokenizer, id2label=id2label,
        max_len=max_len, device=device,
    )

def _fix_bio(tags: List[str]) -> List[str]:
    fixed: List[str] = []
    prev_type: str = ""
    for tag in tags:
        if tag == "O" or tag.startswith("B-"):
            fixed.append(tag)
            prev_type = tag[2:] if tag.startswith("B-") else ""
        elif tag.startswith("I-"):
            cur_type = tag[2:]
            if prev_type == cur_type:
                fixed.append(tag)
            else:
                fixed.append(f"B-{cur_type}")
                prev_type = cur_type
        else:
            fixed.append(tag)
            prev_type = ""
    return fixed

def _majority_vote(per_seed_tags: List[List[str]]) -> List[str]:
    n_tokens = len(per_seed_tags[0])
    voted: List[str] = []
    for i in range(n_tokens):
        counts = Counter(s[i] for s in per_seed_tags)
        top = counts.most_common()
        best_count = top[0][1]
        winners = [lab for lab, c in top if c == best_count]
        if len(winners) == 1:
            voted.append(winners[0])
        elif "O" in winners:
            voted.append("O")
        else:
            voted.append(per_seed_tags[0][i])
    return _fix_bio(voted)

def _predict_ensemble(
    members: List[DirectNERPipeline], tokens_list: Sequence[Sequence[str]],
) -> List[List[str]]:
    per_seed = [
        [m.predict_tokens(list(tokens)) for tokens in tokens_list]
        for m in members
    ]
    voted: List[List[str]] = []
    for i in range(len(tokens_list)):
        voted.append(_majority_vote([per_seed[s][i] for s in range(len(members))]))
    return voted

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, type=str)
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--predict-test", action="store_true",
                    help="Also write submission.csv/submission.zip for the test set.")
    ap.add_argument("--test-path", type=str, default="datasets/new_test_data.csv")
    ap.add_argument("--id-pattern", type=str, default=DEFAULT_ID_PATTERN)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    with open(exp_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    seeds = list(cfg.get("ensemble", {}).get("seeds", [42, 123, 999]))

    print(f"Loading {len(seeds)} ensemble members: seeds={seeds} ...")
    members = [_load_member(exp_dir, s, cfg) for s in seeds]

    split_path = (
        cfg.get("data", {}).get(f"{args.split}_path")
        or f"datasets/new_{args.split}_data.csv"
    )
    df = load_split(split_path)

    print(f"Predicting {len(df)} {args.split} segments with {len(members)}-way ensemble ...")
    tokens_list = [list(t) for t in df["tokens"]]
    preds = _predict_ensemble(members, tokens_list)

    gold = [list(t) for t in df["ner_tags"]]
    metrics = evaluate_predictions(gold, preds, print_report=True)
    out_metrics = exp_dir / f"eval_{args.split}.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {out_metrics}")

    if args.predict_test:
        test_df = load_split(args.test_path, has_labels=False)
        if args.id_pattern:
            pat = re.compile(args.id_pattern)
            mask = test_df["ID"].astype(str).map(lambda s: bool(pat.match(s)))
            kept, dropped = int(mask.sum()), int((~mask).sum())
            if dropped:
                print(
                    f"[filter] Keeping {kept} IDs matching {args.id_pattern!r}; "
                    f"dropping {dropped} rows."
                )
            test_df = test_df[mask].reset_index(drop=True)

        print(f"Predicting {len(test_df)} test segments with ensemble ...")
        test_tokens = [list(t) for t in test_df["tokens"]]
        test_preds = _predict_ensemble(members, test_tokens)

        sub_df = pd.DataFrame({"ID": test_df["ID"], "predicted_ner_tags": test_preds})
        sub_csv = exp_dir / "submission.csv"
        sub_df.to_csv(sub_csv, index=False)
        sub_zip = exp_dir / "submission.zip"
        with zipfile.ZipFile(sub_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(sub_csv, arcname="submission.csv")
        print(f"Wrote {sub_csv} ({len(sub_df)} rows) and {sub_zip}")

if __name__ == "__main__":
    main()
