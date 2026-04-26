from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from src.classifiers.features import SpanFeatureExtractor
from src.classifiers.svm_span import SVMSpanClassifier, fit_svm
from src.data.processor import load_split
from src.models.factory import build_model

Span = Tuple[int, int, str]

def _extract_gold_spans(tags: List[str]) -> List[Span]:
    out: List[Span] = []
    start = None
    cur = None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None and cur is not None:
                out.append((start, i - 1, cur))
            cur = tag[2:]
            start = i
        elif tag.startswith("I-") and cur is not None and tag[2:] == cur:
            continue
        else:
            if start is not None and cur is not None:
                out.append((start, i - 1, cur))
            start, cur = None, None
    if start is not None and cur is not None:
        out.append((start, len(tags) - 1, cur))
    return [(s, e, c) for (s, e, c) in out if c in ("ClinicalImpacts", "SocialImpacts")]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_dir = Path(cfg["output_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    encoder_src_exp = cfg["svm"]["encoder_exp_dir"]
    ckpt = Path(encoder_src_exp) / "checkpoints" / "best" / "model.pt"
    if not ckpt.exists():
        ckpt = Path(encoder_src_exp) / "checkpoints" / "last" / "model.pt"
    print(f"Loading encoder checkpoint: {ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["model_name"])
    model = build_model(cfg["encoder"])
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = SpanFeatureExtractor(
        model=model,
        tokenizer=tokenizer,
        max_len=int(cfg["encoder"].get("max_len", 256)),
        overlap_ratio=float(cfg["svm"].get("overlap_ratio", 0.25)),
        device=device,
    )

    train_path = cfg["data"]["train_path"]
    df = load_split(train_path)
    print(f"Loaded {len(df)} training segments from {train_path}")

    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    t0 = time.time()
    for idx, row in df.iterrows():
        tokens = list(row["tokens"])
        tags = list(row["ner_tags"])
        spans = _extract_gold_spans(tags)
        if not spans:
            continue
        pool = extractor.extract(tokens, [(s, e) for s, e, _ in spans])
        X_list.append(pool)
        y_list.extend(cls for _, _, cls in spans)
        if (idx + 1) % 100 == 0:
            n_spans = sum(x.shape[0] for x in X_list)
            dt = time.time() - t0
            print(f"  [{idx + 1}/{len(df)}] segments  {n_spans} gold spans  {dt:.1f}s")

    X = np.vstack(X_list) if X_list else np.zeros((0, 768), dtype=np.float32)
    y = np.array(y_list, dtype=object)
    print(f"Extracted features: X={X.shape}, y={y.shape}, "
          f"class_counts={dict(zip(*np.unique(y, return_counts=True)))}")

    svm_cfg = cfg["svm"]
    pipeline, report = fit_svm(
        X, y,
        n_pca=int(svm_cfg.get("n_pca", 128)),
        cv_folds=int(svm_cfg.get("cv_folds", 5)),
        C_grid=svm_cfg.get("C_grid"),
        gamma_grid=svm_cfg.get("gamma_grid"),
    )
    print(f"CV macro-F1={report.cv_macro_f1:.4f}  best_params={report.best_params}")

    out_model = exp_dir / "svm.joblib"
    SVMSpanClassifier(pipeline).save(
        out_model,
        meta={
            "encoder_exp_dir": encoder_src_exp,
            "n_pca": int(svm_cfg.get("n_pca", 128)),
            "best_params": report.best_params,
            "cv_macro_f1": report.cv_macro_f1,
            "n_train": report.n_train,
            "class_counts": report.class_counts,
        },
    )
    print(f"Wrote {out_model}")

    np.savez_compressed(
        exp_dir / "gold_span_features.npz", X=X, y=y,
    )

    with open(exp_dir / "svm_train_report.json", "w") as f:
        json.dump({
            "best_params": report.best_params,
            "cv_macro_f1": report.cv_macro_f1,
            "n_train": report.n_train,
            "class_counts": report.class_counts,
        }, f, indent=2)
    print(f"Wrote {exp_dir / 'svm_train_report.json'}")

    dst_ckpt_dir = exp_dir / "checkpoints" / "best"
    if not (dst_ckpt_dir / "model.pt").exists():
        dst_ckpt_dir.mkdir(parents=True, exist_ok=True)
        src_best = Path(encoder_src_exp) / "checkpoints" / "best"
        for f in ["model.pt", "tokenizer.json", "tokenizer_config.json"]:
            src = src_best / f
            if src.exists():
                import shutil
                shutil.copy(src, dst_ckpt_dir / f)
        print(f"Mirrored encoder checkpoint into {dst_ckpt_dir}")

if __name__ == "__main__":
    main()
