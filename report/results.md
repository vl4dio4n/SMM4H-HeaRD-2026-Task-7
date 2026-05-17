| Exp | Pipeline | Encoder | Labels | Loss | CRF | LLRD | LLM / Classifier | Val strict F1 | Val relaxed F1 | Test strict F1 | Test relaxed F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | direct | mental-roberta-base | 5 | CE | No | No | — | 0.3564 | 0.4889 | 0.3675 | 0.4890 |
| 02a | direct | mental-roberta-base | 5 | Weighted CE | No | No | — | 0.3603 | 0.4928 | 0.4096 | 0.5490 |
| 02b | direct | mental-roberta-base | 5 | Focal | No | No | — | 0.3387 | 0.4924 | 0.3362 | 0.5070 |
| 03 | direct | deberta-v3-base | 5 | Focal | No | Yes | — | 0.3446 | 0.5386 | 0.4240 | 0.5580 |
| 04 | direct | deberta-v3-base+CRF | 5 | CRF NLL | Yes | Yes | — | 0.4130 | 0.5714 | 0.4538 | 0.5420 |
| 05a | two\_stage | deberta-v3-base | 3 | Focal | No | Yes | Qwen2.5-7B-4bit | 0.3941 | 0.4230 | 0.4206 | 0.4490 |
| 05b | audit | deberta-v3-base+CRF | 5 | CRF NLL | Yes | Yes | Qwen2.5-7B-4bit | 0.3860 | 0.5407 | 0.4426 | 0.5490 |
| 06 | ensemble (majority) | deberta-v3-base+CRF ×3 | 5 | CRF NLL | Yes | Yes | — | 0.4098 | 0.5649 | 0.4411 | 0.5510 |
| 07a | two\_stage | DeBERTa (3-cls) | 3 | Focal | No | Yes | Qwen2.5-7B + 4-shot | 0.3814 | 0.4309 | 0.4340 | 0.4830 |
| 07b | audit | DeBERTa+CRF | 5 | CRF NLL | Yes | Yes | Qwen2.5-7B + 4-shot | 0.3797 | 0.5369 | 0.4400 | 0.5520 |
| 08a | two\_stage | DeBERTa (3-cls) | 3 | Focal | No | Yes | Gemma-3-12B zero-shot | 0.3824 | 0.4207 | 0.3946 | 0.4180 |
| 08b | audit | DeBERTa+CRF | 5 | CRF NLL | Yes | Yes | Gemma-3-12B zero-shot | 0.3877 | 0.5502 | 0.4000 | 0.4960 |
| 09a | two\_stage | DeBERTa (3-cls) | 3 | Focal | No | Yes | Gemma-3-12B + 4-shot | 0.4076 | 0.4376 | 0.4279 | 0.4440 |
| **09b** | **audit** | **DeBERTa+CRF** | 5 | CRF NLL | Yes | Yes | **Gemma-3-12B + 4-shot** | **0.4255** | **0.5669** | 0.4320 | 0.5230 |
| 10a | svm\_two\_stage | DeBERTa (3-cls) | 3 | Focal | No | Yes | SVM (RBF, PCA-128, CV-F1=0.79) | 0.3571 | 0.3674 | 0.4327 | 0.4350 |
| 10b | svm\_audit | DeBERTa+CRF | 5 | CRF NLL | Yes | Yes | SVM (RBF, PCA-128, CV-F1=1.00) | 0.4065 | 0.5591 | — | — |