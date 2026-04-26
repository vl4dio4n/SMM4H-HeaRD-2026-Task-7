from __future__ import annotations

import os
from typing import Dict, Optional

from .base import BaseLLMClassifier
from .causal import CausalLMClassifier
from .nli import NLIClassifier

def build_llm(cfg: Dict) -> BaseLLMClassifier:
    kind = cfg.get("kind")
    model_name = cfg.get("model_name")

    if kind is None:
        if model_name and "mnli" in model_name.lower():
            kind = "nli"
        else:
            kind = "causal"

    if kind == "nli":
        return NLIClassifier(model_name=model_name or "facebook/bart-large-mnli")

    if kind == "causal":
        hf_token: Optional[str] = cfg.get("hf_token") or os.getenv("HF_ADMIN_READ_TOKEN")
        return CausalLMClassifier(
            model_name=model_name or "Qwen/Qwen2.5-7B-Instruct",
            quantization=cfg.get("quantization", "4bit"),
            hf_token=hf_token,
            max_context_chars=int(cfg.get("max_context_chars", 2000)),
            max_span_chars=int(cfg.get("max_span_chars", 200)),
        )

    raise ValueError(f"Unknown LLM kind: {kind}")
