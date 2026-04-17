"""Evaluation helpers: per-loader metrics and per-pair breakdown."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

from .data import XLMRCodeSwitchDataset, make_collate_fn


def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Return macro-F1 and classification reports for switch and duration tasks."""
    model.eval()
    sw_pred,  sw_true  = [], []
    dur_pred, dur_true = [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            y_sw  = batch["y_switch"].cpu()
            y_dur = batch["y_duration"].cpu()

            sw_logits, dur_logits = model(ids, mask)
            sw_hat  = sw_logits.argmax(-1).cpu()
            dur_hat = dur_logits.argmax(-1).cpu()

            B, L1 = sw_hat.shape
            for b in range(B):
                for t in range(L1):
                    if y_sw[b, t].item() != -100:
                        sw_pred.append(sw_hat[b, t].item())
                        sw_true.append(y_sw[b, t].item())
                    if y_dur[b, t].item() not in (-1, -100):
                        dur_pred.append(dur_hat[b, t].item())
                        dur_true.append(y_dur[b, t].item())

    sw_f1  = f1_score(sw_true, sw_pred, average="macro", zero_division=0)
    dur_f1 = f1_score(dur_true, dur_pred, average="macro", zero_division=0) if dur_true else 0.0

    return {
        "switch_f1":       sw_f1,
        "duration_f1":     dur_f1,
        "switch_report":   classification_report(
            sw_true, sw_pred,
            target_names=["no-switch", "switch"],
            zero_division=0,
        ),
        "duration_report": (
            classification_report(
                dur_true, dur_pred,
                target_names=["small", "medium", "large"],
                zero_division=0,
            )
            if dur_true
            else "No duration predictions."
        ),
    }


def evaluate_per_pair(
    model:       nn.Module,
    all_stats:   dict,
    pairs:       list[tuple],
    tokenizer,
    device:      torch.device,
    batch_size:  int = 32,
    train_ratio: float = 0.8,
    max_len:     int = 128,
    num_workers: int = 2,
) -> dict[str, dict]:
    """Run evaluation independently on each language pair's validation split."""
    results: dict[str, dict] = {}
    collate = make_collate_fn(tokenizer.pad_token_id)

    for lang1, lang2 in pairs:
        key = f"{lang1}-{lang2}"
        if key not in all_stats or not all_stats[key]["processed_samples"]:
            continue
        samples  = all_stats[key]["processed_samples"]
        split_at = int(len(samples) * train_ratio)
        val_ds   = XLMRCodeSwitchDataset(samples[split_at:], tokenizer, max_len)
        if len(val_ds) == 0:
            continue
        ldr = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, collate_fn=collate, num_workers=num_workers,
        )
        results[key] = evaluate(model, ldr, device)

    return results


def print_sigma_summary(
    train_results:    dict[str, dict],
    zeroshot_results: dict[str, dict],
    *,
    headline: str | None = None,
) -> None:
    """Print σ_universality table to stdout."""
    title = headline or "FINAL RESULTS: XLM-R Full FT + Bidirectional"
    print(f"\n{'='*75}")
    print(title)
    print(f"{'='*75}")

    def _section(title: str, results: dict[str, dict]) -> tuple[list, list]:
        print(f"\n── {title} ──")
        print(f"{'Language Pair':<25} {'Switch F1':<14} {'Duration F1':<14}")
        print("-" * 53)
        sw_vals, dur_vals = [], []
        for key, m in results.items():
            print(f"{key:<25} {m['switch_f1']:<14.4f} {m['duration_f1']:<14.4f}")
            sw_vals.append(m["switch_f1"])
            dur_vals.append(m["duration_f1"])
        return sw_vals, dur_vals

    train_sw,  train_dur = _section(f"TRAIN PAIRS ({len(train_results)})",    train_results)
    zs_sw,     zs_dur    = _section(f"ZERO-SHOT PAIRS ({len(zeroshot_results)})", zeroshot_results)

    all_sw  = train_sw + zs_sw
    all_dur = train_dur + zs_dur

    print(f"\n{'='*75}\nσ_UNIVERSALITY SUMMARY\n{'='*75}")
    print(f"{'Subset':<25} {'Sw Mean':<10} {'Sw σ':<10} {'Dur Mean':<10} {'Dur σ':<10}")
    print("-" * 65)
    if train_sw:
        print(f"{'Train pairs':<25} {np.mean(train_sw):<10.4f} {np.std(train_sw):<10.4f} "
              f"{np.mean(train_dur):<10.4f} {np.std(train_dur):<10.4f}")
    if zs_sw:
        print(f"{'Zero-shot pairs':<25} {np.mean(zs_sw):<10.4f} {np.std(zs_sw):<10.4f} "
              f"{np.mean(zs_dur):<10.4f} {np.std(zs_dur):<10.4f}")
    if all_sw:
        print(f"{'All pairs':<25} {np.mean(all_sw):<10.4f} {np.std(all_sw):<10.4f} "
              f"{np.mean(all_dur):<10.4f} {np.std(all_dur):<10.4f}")
    print(f"{'='*75}")
