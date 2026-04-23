"""Dataset loading, preprocessing, label generation, and PyTorch Dataset classes."""
from __future__ import annotations
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .lid import ProductionLID, is_language_neutral_content


# ── Subword alignment ─────────────────────────────────────────────────────────

def align_subwords_to_words(
    words: List[str],
    word_lids: List[str],
    tokenizer,
) -> Tuple[List[str], List[str]]:
    """Expand word-level LID tags to subword token level."""
    assert len(words) == len(word_lids), (
        f"Length mismatch: {len(words)} words vs {len(word_lids)} tags"
    )
    tokens:     List[str] = []
    token_lids: List[str] = []
    for word, lid in zip(words, word_lids):
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            continue
        for token in word_tokens:
            # Both SentencePiece (▁) and BPE (Ġ) word-boundary markers are stripped
            clean = token.replace("▁", "").replace("Ġ", "").strip()
            tag   = "neutral" if (not clean or is_language_neutral_content(clean)) else lid
            tokens.append(token)
            token_lids.append(tag)
    return tokens, token_lids


# ── Label generation ──────────────────────────────────────────────────────────

def generate_labels(
    token_lids: List[str],
) -> Tuple[List[int], List[int]]:
    """
    Build anticipatory switch / duration labels.

    y_switch[t]   = 1 if token t+1 is in a different language, else 0.
    y_duration[t] = burst-length class {0: 1-2, 1: 3-6, 2: 7+} on switch events,
                    -1 on non-switch or neutral positions.
    """
    n = len(token_lids)
    if n < 2:
        return [], []

    # Precompute next non-neutral index for each position
    next_nn: List[Optional[int]] = [None] * n
    last_nn: Optional[int]       = None
    for i in range(n - 1, -1, -1):
        if token_lids[i] != "neutral":
            last_nn = i
        next_nn[i] = last_nn

    y_switch:   List[int] = []
    y_duration: List[int] = []

    for t in range(n - 1):
        lid = token_lids[t]
        if lid == "neutral":
            y_switch.append(0)
            y_duration.append(-1)
            continue

        next_pos = next_nn[t + 1] if t + 1 < n else None
        if next_pos is None:
            y_switch.append(0)
            y_duration.append(-1)
            continue

        next_lid  = token_lids[next_pos]
        is_switch = lid != next_lid
        y_switch.append(1 if is_switch else 0)

        if is_switch:
            burst_len = 1
            for i in range(next_pos + 1, n):
                if token_lids[i] == "neutral":
                    continue
                if token_lids[i] == next_lid:
                    burst_len += 1
                else:
                    break
            if burst_len <= 2:
                y_duration.append(0)
            elif burst_len <= 6:
                y_duration.append(1)
            else:
                y_duration.append(2)
        else:
            y_duration.append(-1)

    return y_switch, y_duration


# ── Sample processing ─────────────────────────────────────────────────────────

def process_sample(
    sample: dict,
    lang1: str,
    lang2: str,
    lid_system: ProductionLID,
    tokenizer,
) -> Optional[dict]:
    """Run LID + label generation for a single dataset example.

    The returned dict stores word-level data (backbone-agnostic) alongside
    subword-level data derived from the given tokenizer.  Downstream code
    can re-tokenize with a different tokenizer using the stored words/word_lids
    without re-running the expensive LID step.
    """
    raw = sample.get("data_generation_result", "")
    if isinstance(raw, list):
        text = " ".join(str(s) for s in raw if s)
    else:
        text = raw if isinstance(raw, str) else ""
    if not text or not text.strip():
        return None
    try:
        words, word_lids         = lid_system.word_level_lid(text, lang1, lang2)
        tokens, token_lids       = align_subwords_to_words(words, word_lids, tokenizer)
        if len(tokens) < 2:
            return None
        y_switch, y_duration     = generate_labels(token_lids)
        return {
            "text":         text,
            # word-level (backbone-agnostic) — used for re-tokenisation
            "words":        words,
            "word_lids":    word_lids,
            # subword-level (tokenizer-specific) — legacy fallback
            "tokens":       tokens,
            "token_lids":   token_lids,
            "y_switch":     y_switch,
            "y_duration":   y_duration,
            "cs_type":      sample.get("cs_type",     "unknown"),
            "cs_function":  sample.get("cs_function", "unknown"),
        }
    except Exception as e:
        print(f"[process_sample] {lang1}-{lang2} | {type(e).__name__}: {e}")
        return None


def analyze_language_pair(
    dataset_split,
    lang1: str,
    lang2: str,
    lid_system: ProductionLID,
    tokenizer,
    max_samples: int = 6000,
) -> Optional[Dict[str, Any]]:
    """Process all samples for one language pair and return aggregate stats."""
    print(f"\n{'='*70}\nProcessing: {lang1} - {lang2}\n{'='*70}")

    filtered = dataset_split.filter(
        lambda x: (
            (x.get("first_language") == lang1 and x.get("second_language") == lang2) or
            (x.get("first_language") == lang2 and x.get("second_language") == lang1)
        )
    )
    print(f"Found {len(filtered)} samples")
    if len(filtered) == 0:
        print("No samples found, skipping.")
        return None

    n        = min(len(filtered), max_samples)
    filtered = filtered.select(range(n))

    stats: Dict[str, Any] = {
        "lang_pair":               f"{lang1}-{lang2}",
        "total_samples":           0,
        "total_tokens":            0,
        "total_switches":          0,
        "duration_distribution":   Counter(),
        "cs_type_distribution":    Counter(),
        "switch_rate_per_sample":  [],
        "processed_samples":       [],
    }

    for idx in tqdm(range(n), desc=f"  {lang1}-{lang2}"):
        result = process_sample(filtered[idx], lang1, lang2, lid_system, tokenizer)
        if result is None:
            continue
        stats["total_samples"]  += 1
        stats["total_tokens"]   += len(result["tokens"])
        stats["total_switches"] += sum(result["y_switch"])
        for dur in result["y_duration"]:
            if dur != -1:
                stats["duration_distribution"][dur] += 1
        stats["cs_type_distribution"][result["cs_type"]] += 1
        if result["tokens"]:
            stats["switch_rate_per_sample"].append(
                sum(result["y_switch"]) / len(result["tokens"])
            )
        stats["processed_samples"].append(result)

    failed = n - stats["total_samples"]
    print(f"✓ Processed: {stats['total_samples']:,}  |  Failed: {failed}")
    if stats["total_tokens"] > 0:
        sr = stats["total_switches"] / stats["total_tokens"]
        print(f"  Tokens: {stats['total_tokens']:,}  |  Switches: {stats['total_switches']:,}  |  Rate: {sr:.2%}")

    return stats


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class CodeSwitchDataset(Dataset):
    """Tokenizer-agnostic code-switching dataset.

    If the sample dict contains 'words'/'word_lids' (produced by the current
    preprocess.py), the dataset re-tokenises on-the-fly with the supplied
    tokenizer so the same pickle works with any backbone.

    Samples that only have 'tokens'/'y_switch'/'y_duration' (legacy pickles
    created before this refactor) are used as-is; they require the same
    tokenizer that was used during preprocessing.
    """

    def __init__(self, samples: list[dict], tokenizer, max_len: int = 128) -> None:
        self.items: list[dict] = []
        for s in samples:
            if "words" in s and "word_lids" in s:
                # New format: re-tokenise with the current tokenizer
                tokens, token_lids = align_subwords_to_words(
                    s["words"], s["word_lids"], tokenizer
                )
                if len(tokens) < 2:
                    continue
                y_switch, y_duration = generate_labels(token_lids)
                ids = tokenizer.convert_tokens_to_ids(tokens)
            else:
                # Legacy format: use stored subword tokens directly
                ids      = tokenizer.convert_tokens_to_ids(s["tokens"])
                y_switch  = s["y_switch"]
                y_duration = s["y_duration"]

            ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]

            sw_raw  = [x if x != -1 else -100 for x in y_switch]
            dur_raw = [x if x != -1 else -100 for x in y_duration]
            sw  = [-100] + sw_raw  + [-100]
            dur = [-100] + dur_raw + [-100]

            ids = ids[:max_len]
            n   = len(ids)
            if n < 2:
                continue

            label_len = n - 1
            sw  = sw[:label_len]
            dur = dur[:label_len]
            if len(sw) != label_len or len(dur) != label_len:
                continue

            self.items.append({
                "input_ids":  torch.tensor(ids, dtype=torch.long),
                "y_switch":   torch.tensor(sw,  dtype=torch.long),
                "y_duration": torch.tensor(dur, dtype=torch.long),
            })

    def __len__(self)        -> int:  return len(self.items)
    def __getitem__(self, i) -> dict: return self.items[i]


# Backward-compat alias
XLMRCodeSwitchDataset = CodeSwitchDataset


def make_collate_fn(pad_id: int):
    """Return a collate function that pads to the longest sequence in a batch."""
    def collate(batch: list[dict]) -> dict:
        ids  = pad_sequence([b["input_ids"]  for b in batch], batch_first=True, padding_value=pad_id)
        sw   = pad_sequence([b["y_switch"]   for b in batch], batch_first=True, padding_value=-100)
        dur  = pad_sequence([b["y_duration"] for b in batch], batch_first=True, padding_value=-100)
        mask = (ids != pad_id).long()
        return {"input_ids": ids, "attention_mask": mask, "y_switch": sw, "y_duration": dur}
    return collate


def build_datasets(
    all_stats: dict,
    pairs: list[tuple],
    tokenizer,
    train_ratio: float = 0.8,
    max_len: int = 128,
) -> Tuple[ConcatDataset, ConcatDataset]:
    """Build concatenated train / val datasets from specified language pairs."""
    train_sets: list[Dataset] = []
    val_sets:   list[Dataset] = []
    print("Building datasets...")
    for lang1, lang2 in pairs:
        key = f"{lang1}-{lang2}"
        if key not in all_stats or not all_stats[key]["processed_samples"]:
            print(f"  ⚠ {key} not found, skipping.")
            continue
        samples  = all_stats[key]["processed_samples"]
        split_at = int(len(samples) * train_ratio)
        tr = CodeSwitchDataset(samples[:split_at], tokenizer, max_len)
        va = CodeSwitchDataset(samples[split_at:],  tokenizer, max_len)
        print(f"  {key:<25}  total={len(samples):,}  train={len(tr):,}  val={len(va):,}")
        train_sets.append(tr)
        val_sets.append(va)
    return ConcatDataset(train_sets), ConcatDataset(val_sets)
