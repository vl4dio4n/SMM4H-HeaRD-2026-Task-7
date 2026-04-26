from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import SMM4HDataset, build_collate_fn
from src.data.processor import (
    LABELS_3,
    LABELS_5,
    collapse_5_to_3,
    label_to_id,
    load_split,
)
from src.models.factory import build_model
from src.training.losses import build_loss
from src.training.trainer import Trainer

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)

def prepare_data(cfg: Dict, tokenizer):
    enc_cfg = cfg["encoder"]
    num_labels = int(enc_cfg["num_labels"])
    max_len = int(enc_cfg.get("max_len", 256))

    if num_labels == 5:
        labels = LABELS_5
        tag_transform = lambda t: t
    elif num_labels == 3:
        labels = LABELS_3
        tag_transform = collapse_5_to_3
    else:
        raise ValueError(f"num_labels must be 3 or 5, got {num_labels}")

    l2id = label_to_id(labels)
    data_cfg = cfg.get("data", {})
    train_path = data_cfg.get("train_path", "datasets/new_train_data.csv")
    dev_path = data_cfg.get("dev_path", "datasets/new_dev_data.csv")

    df_train = load_split(train_path)
    df_dev = load_split(dev_path)

    train_tags = [tag_transform(list(t)) for t in df_train["ner_tags"]]
    dev_tags = [tag_transform(list(t)) for t in df_dev["ner_tags"]]

    train_ds = SMM4HDataset(
        tokens_list=df_train["tokens"].tolist(),
        tags_list=train_tags,
        tokenizer=tokenizer,
        label2id=l2id,
        max_len=max_len,
        ids=df_train["ID"].astype(str).tolist(),
    )
    val_ds = SMM4HDataset(
        tokens_list=df_dev["tokens"].tolist(),
        tags_list=dev_tags,
        tokenizer=tokenizer,
        label2id=l2id,
        max_len=max_len,
        ids=df_dev["ID"].astype(str).tolist(),
    )
    return train_ds, val_ds, labels, l2id

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--subset", type=int, default=0,
                    help="(debug) train on the first N rows only; 0 = full")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("training", {}).get("seed", 42)))

    output_dir = Path(cfg.get("output_dir") or f"out/experiments/{cfg['experiment_name']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["model_name"])
    train_ds, val_ds, labels, l2id = prepare_data(cfg, tokenizer)

    if args.subset:
        train_ds.tokens_list = train_ds.tokens_list[: args.subset]
        train_ds.tags_list = train_ds.tags_list[: args.subset]
        if train_ds.ids is not None:
            train_ds.ids = train_ds.ids[: args.subset]
        val_ds.tokens_list = val_ds.tokens_list[: max(4, args.subset // 4)]
        val_ds.tags_list = val_ds.tags_list[: max(4, args.subset // 4)]
        if val_ds.ids is not None:
            val_ds.ids = val_ds.ids[: max(4, args.subset // 4)]
        print(f"[debug] subset: train={len(train_ds)}, val={len(val_ds)}")

    collate = build_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(cfg["training"].get("num_workers", 2)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])),
        shuffle=False,
        collate_fn=collate,
        num_workers=int(cfg["training"].get("num_workers", 2)),
    )

    model = build_model(cfg["encoder"])
    if cfg["encoder"].get("use_crf", False):
        loss_fn = None
    else:
        loss_fn = build_loss(cfg.get("loss", {"type": "ce"}))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    id2label = {i: lab for lab, i in l2id.items()}

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        id2label=id2label,
        output_dir=output_dir,
        cfg=cfg,
        device=device,
    )
    state = trainer.train()
    print(
        f"Done. Best val strict F1 = {state.best_f1:.4f} at epoch {state.best_epoch}.\n"
        f"Artifacts: {output_dir.resolve()}"
    )

if __name__ == "__main__":
    main()
