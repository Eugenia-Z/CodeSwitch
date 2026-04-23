"""Visualization: universality bar chart, training-history curves, and analysis plots."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def plot_universality(
    train_results:    dict[str, dict],
    zeroshot_results: dict[str, dict],
    output_path:      str = "results/xlmr_universality.png",
    title:            str = "Anticipatory F1 Across Language Pairs",
) -> None:
    """Bar chart of switch / duration F1 for all language pairs."""
    all_results = {**train_results, **zeroshot_results}
    pair_names  = list(train_results.keys()) + list(zeroshot_results.keys())
    sw_vals     = [all_results[p]["switch_f1"]   for p in pair_names]
    dur_vals    = [all_results[p]["duration_f1"] for p in pair_names]
    colors      = ["#4C72B0"] * len(train_results) + ["#DD8452"] * len(zeroshot_results)
    x           = np.arange(len(pair_names))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, vals, ylabel, title, mean_color in [
        (axes[0], sw_vals,  "Switch F1 (macro)",   "Switch F1",   "blue"),
        (axes[1], dur_vals, "Duration F1 (macro)", "Duration F1", "orange"),
    ]:
        bars = ax.bar(x, vals, color=colors, alpha=0.85)
        ax.axhline(
            np.mean(vals), color=mean_color, linestyle="--",
            linewidth=1, alpha=0.6, label=f"Mean={np.mean(vals):.3f}",
        )
        if ax is axes[0]:
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.4, label="Random")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_names, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.legend(
        handles=[
            Patch(facecolor="#4C72B0", alpha=0.85, label="Train pairs"),
            Patch(facecolor="#DD8452", alpha=0.85, label="Zero-shot pairs"),
        ],
        loc="upper right", fontsize=9,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.show()


def plot_training_history(
    history:     list[dict],
    output_path: str = "results/training_history.png",
) -> None:
    """Loss and F1 curves across training epochs."""
    epochs   = [h["epoch"]       for h in history]
    sw_f1s   = [h["switch_f1"]   for h in history]
    dur_f1s  = [h["duration_f1"] for h in history]
    loss_sw  = [h["loss_sw"]     for h in history]
    loss_dur = [h["loss_dur"]    for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, loss_sw,  label="Switch loss",   marker="o")
    axes[0].plot(epochs, loss_dur, label="Duration loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, sw_f1s,  label="Switch F1",   marker="o")
    axes[1].plot(epochs, dur_f1s, label="Duration F1", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.show()


def plot_per_pair_training_curves(
    history:     list[dict],
    pairs:       list[str],
    output_path: str = "results/per_pair_curves.png",
) -> None:
    """Grid of per-pair switch-F1 curves from training history."""
    n    = len(pairs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    epochs = [h["epoch"] for h in history]

    for idx, pair_key in enumerate(pairs):
        ax = axes_flat[idx]
        f1s = [h["pair_f1s"].get(pair_key, 0.0) for h in history]
        ax.plot(epochs, f1s, marker="o", linewidth=1.5)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.5, label="Random")
        ax.set_title(pair_key, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Switch F1", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Per-Pair Switch F1 During Training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.show()


def plot_grouped_f1_bars(
    results:     dict[str, dict],
    output_path: str = "results/grouped_f1.png",
    title:       str = "Switch F1 vs Duration F1 per Language Pair",
) -> None:
    """Paired bars comparing switch F1 and duration F1 per language pair."""
    pairs    = list(results.keys())
    sw_vals  = [results[p]["switch_f1"]   for p in pairs]
    dur_vals = [results[p]["duration_f1"] for p in pairs]
    x        = np.arange(len(pairs))
    w        = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(pairs) * 0.9), 5))
    ax.bar(x - w / 2, sw_vals,  w, label="Switch F1",   color="#4C72B0", alpha=0.85)
    ax.bar(x + w / 2, dur_vals, w, label="Duration F1", color="#DD8452", alpha=0.85)
    ax.axhline(np.mean(sw_vals),  color="#4C72B0", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(np.mean(dur_vals), color="#DD8452", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Macro F1")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.show()


def plot_lr_schedule(
    warmup_steps: int,
    total_steps:  int,
    base_lr:      float = 1e-5,
    head_lr:      float = 5e-4,
    output_path:  str   = "results/lr_schedule.png",
) -> None:
    """Visualize the warmup + cosine LR schedule for encoder and heads."""
    steps = np.arange(total_steps)

    def _cosine_warmup(s: int) -> float:
        if s < warmup_steps:
            return s / max(warmup_steps, 1)
        prog = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * prog)))

    factors  = np.array([_cosine_warmup(int(s)) for s in steps])
    enc_lrs  = base_lr * factors
    head_lrs = head_lr * factors

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, enc_lrs,  label=f"Encoder LR (peak={base_lr:.0e})")
    ax.plot(steps, head_lrs, label=f"Head LR (peak={head_lr:.0e})", linestyle="--")
    ax.axvline(warmup_steps, color="grey", linestyle=":", alpha=0.7, label="Warmup end")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR Schedule: Linear Warmup + Cosine Decay", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.show()
