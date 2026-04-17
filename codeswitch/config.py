from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

# ── Language pairs ────────────────────────────────────────────────────────────

ALL_LANGUAGE_PAIRS: List[Tuple[str, str]] = [
    ("Cantonese",   "English"),
    ("Arabic",      "English"),
    ("Philippines", "English"),   # dataset key (typo "Philipines" corrected)
    ("German",      "French"),
    ("Chinese",     "English"),
    ("Vietnamese",  "English"),
    ("Malay",       "English"),
    ("Japanese",    "English"),
    ("Hindi",       "English"),
    ("Korean",      "English"),
    ("Spanish",     "English"),
    ("French",      "English"),
    ("Russian",     "English"),
    ("Italian",     "English"),
    ("German",      "English"),
]

# Default: train on all 15 pairs (no zero-shot split)
TRAIN_PAIRS: List[Tuple[str, str]] = [
    ("Chinese",     "English"),
    ("Hindi",       "English"),
    ("Italian",     "English"),
    ("German",      "English"),
    ("Arabic",      "English"),
    ("Japanese",    "English"),
    ("Vietnamese",  "English"),
    ("Spanish",     "English"),
    ("Korean",      "English"),
    ("Russian",     "English"),
    ("French",      "English"),
    ("Malay",       "English"),
    ("Philippines", "English"),
    ("German",      "French"),
    ("Cantonese",   "English"),
]

ZEROSHOT_PAIRS: List[Tuple[str, str]] = []


def parse_pair(pair_str: str) -> Tuple[str, str]:
    """Parse 'Lang1-Lang2' CLI string into a (Lang1, Lang2) tuple."""
    parts = pair_str.split("-", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid pair format: {pair_str!r}. Expected 'Lang1-Lang2'."
        )
    return parts[0].strip(), parts[1].strip()


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    model_name:     str   = "xlm-roberta-base"
    lid_model:      str   = "papluca/xlm-roberta-base-language-detection"
    dropout:        float = 0.1
    freeze_encoder: bool  = False
    max_len:        int   = 128
    # When True: training stays full-sequence bidirectional; in eval() only, use
    # prefix encodes so hidden[t] never attends to tokens > t (causal for t→t+1).
    use_causal_at_eval: bool = False


@dataclass
class DataConfig:
    dataset_name:        str   = "Shelton1013/SwitchLingua_text"
    max_samples_per_pair: int  = 6000
    train_ratio:         float = 0.8


@dataclass
class TrainConfig:
    batch_size:        int   = 32
    num_epochs:        int   = 8
    base_lr:           float = 2e-5
    head_lr_multiplier: float = 50.0
    weight_decay:      float = 0.01
    grad_clip:         float = 1.0
    lambda_dur:        float = 1.0
    seed:              int   = 42
    num_workers:       int   = 2

    @property
    def head_lr(self) -> float:
        return self.base_lr * self.head_lr_multiplier
