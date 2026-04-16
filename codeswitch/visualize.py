"""Visualization: universality bar chart and training-history curves."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def plot_universality(
    train_results:    dict[str, dict],
    zeroshot_results: dict[str, dict],
    output_path:      str = "results/xlmr_universality.png",
) -> None:
    """Bar chart of switch / duration F1 for all language pairs."""
    all_results = {**train_results, **zeroshot_results}
    pair_names  = list(train_results.keys()) + list(zeroshot_results.keys())
    sw_vals     = [all_results[p]["switch_f1"]   for p in pair_names]
    dur_vals    = [all_results[p]["duration_f1"] for p in pair_names]
    colors      = ["#4C72B0"] * len(train_results) + ["#DD8452"] * len(zeroshot_results)
    x           = np.arange(len(pair_names))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "XLM-R Full FT: Anticipatory F1 Across Language Pairs",
        fontsize=14, fontweight="bold",
    )

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
