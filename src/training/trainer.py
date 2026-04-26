from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from ..evaluation.metrics import evaluate_predictions
from ..models.crf_encoder import CRFEncoder
from ..models.encoder import TokenClassificationEncoder

@dataclass
class TrainState:
    best_f1: float = -1.0
    best_epoch: int = -1
    history: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

def _build_llrd_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    decay: float,
    weight_decay: float = 0.01,
) -> List[Dict]:
    if isinstance(model, CRFEncoder):
        backbone = model.backbone.backbone
        head_params = [
            (n, p) for n, p in model.named_parameters()
            if not n.startswith("backbone.backbone.")
        ]
    else:
        backbone = model.backbone
        head_params = [
            (n, p) for n, p in model.named_parameters()
            if not n.startswith("backbone.")
        ]

    num_layers = getattr(backbone.config, "num_hidden_layers", 12)
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    groups: List[Dict] = []

    def wd(name: str) -> float:
        return 0.0 if any(nd in name for nd in no_decay) else weight_decay

    for name, param in head_params:
        if not param.requires_grad:
            continue
        groups.append({"params": [param], "lr": base_lr, "weight_decay": wd(name)})

    layer_lrs: Dict[int, float] = {}
    for i in range(num_layers):
        layer_lrs[i] = base_lr * (decay ** (num_layers - i))

    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue
        layer_idx: Optional[int] = None
        parts = name.split(".")
        for p_idx, part in enumerate(parts):
            if part in {"layer", "encoder"} and p_idx + 1 < len(parts) and parts[p_idx + 1].isdigit():
                layer_idx = int(parts[p_idx + 1])
                break
        if layer_idx is None:
            lr = base_lr * (decay ** (num_layers + 1))
        else:
            lr = layer_lrs[layer_idx]
        groups.append({"params": [param], "lr": lr, "weight_decay": wd(name)})
    return groups

def _build_optimizer(
    model: torch.nn.Module, optim_cfg: Dict
) -> torch.optim.Optimizer:
    base_lr = float(optim_cfg.get("base_lr", 2e-5))
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))
    use_llrd = bool(optim_cfg.get("use_llrd", False))
    if use_llrd:
        groups = _build_llrd_param_groups(
            model,
            base_lr=base_lr,
            decay=float(optim_cfg.get("llrd_decay", 0.9)),
            weight_decay=weight_decay,
        )
        return AdamW(groups, lr=base_lr)
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay_params if any(nd in n for nd in no_decay) else decay_params).append(p)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=base_lr)

def _decode_preds(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    id2label: Dict[int, str],
    device: str,
) -> Tuple[List[List[str]], List[List[str]]]:
    is_crf = isinstance(model, CRFEncoder)
    word_ids = batch["word_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    if is_crf:
        best_paths, word_indices = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=attn,
            word_ids=word_ids,
        )
    else:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=attn,
            token_type_ids=batch.get("token_type_ids").to(device) if batch.get("token_type_ids") is not None else None,
        )
        preds = logits.argmax(-1)

    B = attn.size(0)
    num_words = batch["num_words"].tolist()
    y_pred_batch: List[List[str]] = []
    y_true_batch: List[List[str]] = []

    for b in range(B):
        n_words = int(num_words[b])
        pred_words: List[str] = ["O"] * n_words
        if is_crf:
            for p_idx, wid in enumerate(word_indices[b]):
                if 0 <= wid < n_words:
                    pred_words[wid] = id2label[best_paths[b][p_idx]]
        else:
            prev_wid = -2
            for t in range(attn.size(1)):
                wid = int(word_ids[b, t].item())
                if wid < 0 or wid >= n_words:
                    continue
                if wid != prev_wid:
                    pred_words[wid] = id2label[int(preds[b, t].item())]
                    prev_wid = wid

        true_words: List[str] = ["O"] * n_words
        prev_wid = -2
        for t in range(attn.size(1)):
            wid = int(word_ids[b, t].item())
            if wid < 0 or wid >= n_words:
                continue
            lab = int(labels[b, t].item())
            if lab == -100:
                continue
            if wid != prev_wid:
                true_words[wid] = id2label[lab]
                prev_wid = wid

        y_pred_batch.append(pred_words)
        y_true_batch.append(true_words)
    return y_true_batch, y_pred_batch

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        loss_fn: Optional[torch.nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        id2label: Dict[int, str],
        output_dir: str | os.PathLike,
        cfg: Dict,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn.to(device) if loss_fn is not None else None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.id2label = id2label
        self.cfg = cfg
        self.device = device

        self.output_dir = Path(output_dir)
        (self.output_dir / "checkpoints" / "best").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints" / "last").mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        tcfg = cfg.get("training", {})
        self.epochs = int(tcfg.get("epochs", 10))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.amp_dtype = torch.bfloat16 if tcfg.get("bf16", True) else torch.float32

        self.optimizer = _build_optimizer(self.model, cfg.get("optimizer", {}))

        total_steps = max(1, len(train_loader) * self.epochs)
        warmup = int(total_steps * float(tcfg.get("warmup_ratio", 0.1)))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
        )
        self.state = TrainState()
        self.is_crf = isinstance(model, CRFEncoder)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
            if self.is_crf:
                loss, _ = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                    word_ids=inputs["word_ids"],
                )
            else:
                logits = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs.get("token_type_ids"),
                )
                loss = self.loss_fn(logits, inputs["labels"])

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return float(loss.detach().item())

    @torch.inference_mode()
    def _validate(self) -> Dict[str, Any]:
        self.model.eval()
        all_true: List[List[str]] = []
        all_pred: List[List[str]] = []
        total_loss, total_batches = 0.0, 0

        for batch in self.val_loader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                if self.is_crf:
                    loss, _ = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"],
                        word_ids=inputs["word_ids"],
                    )
                else:
                    logits = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs.get("token_type_ids"),
                    )
                    loss = self.loss_fn(logits, inputs["labels"])
                total_loss += float(loss.item())
                total_batches += 1

                y_true, y_pred = _decode_preds(self.model, batch, self.id2label, self.device)
                all_true.extend(y_true)
                all_pred.extend(y_pred)

        metrics = evaluate_predictions(all_true, all_pred, print_report=False)
        val_loss = total_loss / max(1, total_batches)
        return {
            "val_loss": val_loss,
            "f1_strict": metrics["strict"]["f1_strict"],
            "precision_strict": metrics["strict"]["precision_strict"],
            "recall_strict": metrics["strict"]["recall_strict"],
            "f1_relaxed": metrics["relaxed"].get("Overall", {}).get("f1", 0.0),
        }

    def _save_checkpoint(self, tag: str) -> None:
        ckpt_dir = self.output_dir / "checkpoints" / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        try:
            self.tokenizer.save_pretrained(str(ckpt_dir))
        except Exception:
            pass

    def train(self) -> TrainState:
        for epoch in range(1, self.epochs + 1):
            losses: List[float] = []
            for batch in self.train_loader:
                losses.append(self._train_step(batch))
            train_loss = sum(losses) / max(1, len(losses))
            val = self._validate()

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **val,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.state.history.append(row)
            with open(self.output_dir / "metrics.json", "w") as f:
                json.dump(self.state.history, f, indent=2)

            is_best = val["f1_strict"] > self.state.best_f1
            print(
                f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
                f"val_loss={val['val_loss']:.4f} "
                f"strict_F1={val['f1_strict']:.4f} "
                f"relaxed_F1={val['f1_relaxed']:.4f}"
                + ("  *BEST*" if is_best else "")
            )
            self._save_checkpoint("last")
            if is_best:
                self.state.best_f1 = val["f1_strict"]
                self.state.best_epoch = epoch
                self._save_checkpoint("best")

        with open(self.output_dir / "best.json", "w") as f:
            json.dump(
                {"best_f1": self.state.best_f1, "best_epoch": self.state.best_epoch},
                f,
                indent=2,
            )
        return self.state
