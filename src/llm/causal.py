from __future__ import annotations

import gc
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base import CLASSES, BaseLLMClassifier, FewShotExample

SYSTEM_PROMPT = (
    "You classify the *type* of a substance-use impact span inside a Reddit post. "
    "Answer with exactly ONE word from {ClinicalImpacts, SocialImpacts, Neither}. "
    "No other text."
)

USER_TEMPLATE_BASE = (
    "ClinicalImpacts = physical/medical/mental-health impacts of substance use "
    "(withdrawal, overdose, addiction, detox, relapse, psychiatric effects, medical treatment).\n"
    "SocialImpacts   = relational/functional impacts "
    "(job loss, homelessness, broken relationships, legal trouble, financial ruin, family conflict).\n"
    "Neither         = the span is not actually an impact of substance use.\n\n"
    "{examples_block}"
    "Post: \"{post}\"\n"
    "Span: \"{span}\"\n"
    "{hint_line}"
    "Answer:"
)

def _normalise(raw: str) -> str:
    s = raw.strip()
    upper = s.upper()
    for cls in CLASSES:
        if cls.upper() in upper:
            return cls
    lower = s.lower()
    if "clinical" in lower:
        return "ClinicalImpacts"
    if "social" in lower:
        return "SocialImpacts"
    return "Neither"

def _format_examples(examples: Optional[List[FewShotExample]]) -> str:
    if not examples:
        return ""
    lines = ["Examples:"]
    for ex in examples:
        lines.append(f"Post: \"{ex.post}\"")
        lines.append(f"Span: \"{ex.span}\"")
        lines.append(f"Answer: {ex.label}")
        lines.append("")
    return "\n".join(lines) + "\n"

class CausalLMClassifier(BaseLLMClassifier):
    def __init__(
        self,
        model_name: str,
        quantization: str = "4bit",
        hf_token: Optional[str] = None,
        max_context_chars: int = 2000,
        max_span_chars: int = 200,
    ) -> None:
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.max_span_chars = max_span_chars

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        load_kwargs: dict = {"token": hf_token, "device_map": "cuda"}
        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

    def _build_prompt(
        self,
        post: str,
        span: str,
        current_label: Optional[str],
        examples: Optional[List[FewShotExample]],
    ) -> str:
        post_trim = post[: self.max_context_chars]
        span_trim = span[: self.max_span_chars]
        hint = (
            f"Encoder's current guess: {current_label}.\n"
            if current_label and current_label in CLASSES
            else ""
        )
        user = USER_TEMPLATE_BASE.format(
            examples_block=_format_examples(examples),
            post=post_trim,
            span=span_trim,
            hint_line=hint,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def _generate(self, prompts: List[str]) -> List[str]:
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def classify_span(
        self,
        post: str,
        span: str,
        current_label: Optional[str] = None,
        examples: Optional[List[FewShotExample]] = None,
    ) -> str:
        prompt = self._build_prompt(post, span, current_label, examples)
        decoded = self._generate([prompt])
        return _normalise(decoded[0])

    def classify_batch(
        self,
        inputs: List[Tuple[str, str, Optional[str]]],
        examples: Optional[List[FewShotExample]] = None,
        batch_size: int = 8,
    ) -> List[str]:
        results: List[str] = []
        for i in range(0, len(inputs), batch_size):
            chunk = inputs[i:i + batch_size]
            prompts = [
                self._build_prompt(p, s, cur, examples) for p, s, cur in chunk
            ]
            decoded = self._generate(prompts)
            results.extend(_normalise(d) for d in decoded)
        return results

    def close(self) -> None:
        try:
            del self.model
        except AttributeError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
