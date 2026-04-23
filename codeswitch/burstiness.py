"""Burstiness analysis: recall of switch events in burst vs isolated regions.

Compares multitask learning (λ_dur=1.0) against single-task (λ_dur=0.0) to measure
whether the duration auxiliary task improves recall in bursty code-switching regions.

A switch event at position t is "bursty" if there are ≥ burst_threshold other switch
events within ±window_size tokens of t (in the ground-truth labels).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .config import BurstinessConfig
from .data import align_subwords_to_words, generate_labels


def _build_switch_sequence(sample: dict, tokenizer, max_len: int) -> Tuple[list, list] | None:
    """Return (input_ids, sw_labels) for one sample, accounting for BOS/EOS truncation."""
    if "words" in sample and "word_lids" in sample:
        tokens, token_lids = align_subwords_to_words(
            sample["words"], sample["word_lids"], tokenizer
        )
        if len(tokens) < 2:
            return None
        y_switch, _ = generate_labels(token_lids)
        ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        ids      = tokenizer.convert_tokens_to_ids(sample["tokens"])
        y_switch = sample["y_switch"]

    ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
    ids = ids[:max_len]
    n   = len(ids)
    if n < 2:
        return None

    label_len = n - 1
    sw_padded = [-100] + [x if x != -1 else -100 for x in y_switch] + [-100]
    sw_padded = sw_padded[:label_len]
    if len(sw_padded) != label_len:
        return None
    return ids, sw_padded


def compute_burstiness_metrics(
    model:      nn.Module,
    all_stats:  dict,
    pairs:      list[tuple],
    tokenizer,
    device:     torch.device,
    cfg:        BurstinessConfig | None = None,
) -> Dict[str, dict]:
    """Compute per-pair bursty / isolated switch recall for one model.

    Returns a dict keyed by pair name with keys:
        bursty_recall, isolated_recall, bursty_n, isolated_n
    """
    if cfg is None:
        cfg = BurstinessConfig()

    model.eval()
    results: Dict[str, dict] = {}
    pad_id = tokenizer.pad_token_id

    for lang1, lang2 in pairs:
        key = f"{lang1}-{lang2}"
        if key not in all_stats or not all_stats[key]["processed_samples"]:
            continue

        samples  = all_stats[key]["processed_samples"]
        split_at = int(len(samples) * cfg.train_ratio)
        val_samp = samples[split_at:]

        bursty_tp = bursty_fn = isolated_tp = isolated_fn = 0

        with torch.no_grad():
            for s in val_samp:
                out = _build_switch_sequence(s, tokenizer, cfg.max_len)
                if out is None:
                    continue
                ids_list, sw = out
                ids_t = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0).to(device)
                mask  = (ids_t != pad_id).long()

                sw_logits, _ = model(ids_t, mask)
                sw_pred = sw_logits[0].argmax(-1).cpu().tolist()

                for t in range(len(sw)):
                    if sw[t] != 1:
                        continue
                    # count other true switches in the window
                    start = max(0, t - cfg.window_size)
                    end   = min(len(sw), t + cfg.window_size + 1)
                    window_count = sum(
                        1 for j in range(start, end)
                        if j != t and j < len(sw) and sw[j] == 1
                    )
                    is_bursty = window_count >= cfg.burst_threshold
                    correct   = t < len(sw_pred) and sw_pred[t] == 1

                    if is_bursty:
                        bursty_tp  += int(correct)
                        bursty_fn  += int(not correct)
                    else:
                        isolated_tp += int(correct)
                        isolated_fn += int(not correct)

        denom_b = bursty_tp   + bursty_fn
        denom_i = isolated_tp + isolated_fn
        results[key] = {
            "bursty_recall":   bursty_tp  / denom_b if denom_b else 0.0,
            "isolated_recall": isolated_tp / denom_i if denom_i else 0.0,
            "bursty_n":   denom_b,
            "isolated_n": denom_i,
        }
        print(
            f"  {key:<25}  bursty_n={denom_b:,}  "
            f"bursty_recall={results[key]['bursty_recall']:.3f}  "
            f"isolated_recall={results[key]['isolated_recall']:.3f}"
        )

    return results


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_burstiness_comparison(
    mt_results: Dict[str, dict],
    st_results: Dict[str, dict],
    output_path: str = "results/burstiness_comparison.png",
) -> None:
    """Two subplots: bursty recall and isolated recall — multitask vs single-task."""
    pairs     = [p for p in mt_results if p in st_results]
    mt_bursty = [mt_results[p]["bursty_recall"]   for p in pairs]
    st_bursty = [st_results[p]["bursty_recall"]   for p in pairs]
    mt_iso    = [mt_results[p]["isolated_recall"] for p in pairs]
    st_iso    = [st_results[p]["isolated_recall"] for p in pairs]
    x = np.arange(len(pairs))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Multitask vs Single-Task: Switch Recall by Region Type", fontsize=13, fontweight="bold")

    for ax, mt_vals, st_vals, ylabel in [
        (axes[0], mt_bursty, st_bursty, "Bursty Region Recall"),
        (axes[1], mt_iso,    st_iso,    "Isolated Region Recall"),
    ]:
        ax.bar(x - w / 2, mt_vals, w, label="Multitask (λ=1.0)", color="#4C72B0", alpha=0.85)
        ax.bar(x + w / 2, st_vals, w, label="Single-task (λ=0.0)", color="#DD8452", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.show()


def plot_burst_delta(
    mt_results:  Dict[str, dict],
    st_results:  Dict[str, dict],
    output_path: str = "results/burstiness_delta.png",
) -> None:
    """Δ recall (MT − ST) for bursty and isolated regions per pair."""
    pairs      = [p for p in mt_results if p in st_results]
    delta_b    = [mt_results[p]["bursty_recall"]   - st_results[p]["bursty_recall"]   for p in pairs]
    delta_i    = [mt_results[p]["isolated_recall"] - st_results[p]["isolated_recall"] for p in pairs]
    x = np.arange(len(pairs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w / 2, delta_b, w, label="Δ Bursty recall",   color="#4C72B0", alpha=0.85)
    ax.bar(x + w / 2, delta_i, w, label="Δ Isolated recall", color="#DD8452", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Recall gain (MT − ST)")
    ax.set_title("Multitask Gain: Bursty vs Isolated Switch Recall", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.show()


def plot_density_recall_scatter(
    mt_results:  Dict[str, dict],
    all_stats:   dict,
    output_path: str = "results/burstiness_density_scatter.png",
) -> None:
    """Scatter: switch density (switches/token) vs bursty/isolated recall per pair."""
    pairs      = list(mt_results.keys())
    densities  = []
    for p in pairs:
        stats = all_stats.get(p, {})
        if stats.get("total_tokens", 0) > 0:
            densities.append(stats["total_switches"] / stats["total_tokens"])
        else:
            densities.append(0.0)

    bursty_vals   = [mt_results[p]["bursty_recall"]   for p in pairs]
    isolated_vals = [mt_results[p]["isolated_recall"] for p in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Switch Density vs Recall (Multitask Model)", fontsize=12, fontweight="bold")

    for ax, vals, title in [
        (axes[0], bursty_vals,   "Bursty Region Recall"),
        (axes[1], isolated_vals, "Isolated Region Recall"),
    ]:
        ax.scatter(densities, vals, alpha=0.8)
        for d, v, p in zip(densities, vals, pairs):
            ax.annotate(p, (d, v), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("Switch density (switches / token)")
        ax.set_ylabel("Recall")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.show()


def print_burstiness_table(
    mt_results: Dict[str, dict],
    st_results: Dict[str, dict] | None = None,
) -> None:
    """Print a per-pair burstiness summary table."""
    pairs = list(mt_results.keys())
    print(f"\n{'='*80}")
    print("BURSTINESS ANALYSIS")
    print(f"{'='*80}")
    if st_results:
        print(f"{'Pair':<25} {'MT Bursty':>10} {'ST Bursty':>10} {'Δ Bursty':>10} "
              f"{'MT Iso':>10} {'ST Iso':>10} {'Δ Iso':>8}")
        print("-" * 80)
        for p in pairs:
            if p not in st_results:
                continue
            mb = mt_results[p]["bursty_recall"];   sb = st_results[p]["bursty_recall"]
            mi = mt_results[p]["isolated_recall"]; si = st_results[p]["isolated_recall"]
            print(f"{p:<25} {mb:>10.3f} {sb:>10.3f} {mb-sb:>+10.3f} "
                  f"{mi:>10.3f} {si:>10.3f} {mi-si:>+8.3f}")
    else:
        print(f"{'Pair':<25} {'Bursty N':>10} {'Bursty Recall':>15} {'Iso N':>8} {'Iso Recall':>12}")
        print("-" * 72)
        for p in pairs:
            r = mt_results[p]
            print(f"{p:<25} {r['bursty_n']:>10,} {r['bursty_recall']:>15.3f} "
                  f"{r['isolated_n']:>8,} {r['isolated_recall']:>12.3f}")
    print(f"{'='*80}")
