"""Qualitative per-token error analysis grouped by code-switch type.

Evaluates a trained model on per-token predictions and breaks down errors by
the cs_type field (inter-sentential, intra-sentential, tag, etc.) so we can
see which types of code-switching are hardest to predict.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from .config import QualitativeConfig
from .data import CodeSwitchDatasetWithMeta, align_subwords_to_words, generate_labels


# ── Per-token evaluation ──────────────────────────────────────────────────────

def evaluate_per_token(
    model:     nn.Module,
    dataset:   CodeSwitchDatasetWithMeta,
    tokenizer,
    device:    torch.device,
) -> List[dict]:
    """Return a list of per-token prediction records for all items in dataset.

    Each record contains sample_idx, token_pos, sw_true, sw_pred, sw_prob,
    dur_true, dur_pred, cs_type, cs_function.
    """
    model.eval()
    pad_id  = tokenizer.pad_token_id
    records = []

    with torch.no_grad():
        for i, item in enumerate(dataset):
            ids_t  = item["input_ids"].unsqueeze(0).to(device)
            mask   = (ids_t != pad_id).long()

            sw_logits, dur_logits = model(ids_t, mask)
            sw_probs  = torch.softmax(sw_logits[0], dim=-1).cpu()   # [L-1, 2]
            sw_pred   = sw_logits[0].argmax(-1).cpu()               # [L-1]
            dur_pred  = dur_logits[0].argmax(-1).cpu()              # [L-1]

            y_sw  = item["y_switch"]
            y_dur = item["y_duration"]

            for t in range(len(y_sw)):
                sw_t = y_sw[t].item()
                if sw_t == -100:
                    continue
                records.append({
                    "sample_idx":   i,
                    "token_pos":    t,
                    "sw_true":      sw_t,
                    "sw_pred":      sw_pred[t].item() if t < len(sw_pred) else -1,
                    "sw_prob":      sw_probs[t, 1].item() if t < len(sw_probs) else 0.0,
                    "dur_true":     y_dur[t].item(),
                    "dur_pred":     dur_pred[t].item() if t < len(dur_pred) else -1,
                    "cs_type":      item.get("cs_type",     "unknown"),
                    "cs_function":  item.get("cs_function", "unknown"),
                })

    return records


def build_qualitative_df(records: List[dict]) -> pd.DataFrame:
    """Convert per-token records to a DataFrame with confusion categories."""
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Normalize cs_type to a small set of canonical categories
    def _normalize_type(t: str) -> str:
        t = str(t).lower()
        if "inter" in t:
            return "inter-sentential"
        if "intra" in t or "mix" in t:
            return "intra-sentential"
        if "tag" in t:
            return "tag"
        return "other"

    df["cs_type_norm"] = df["cs_type"].apply(_normalize_type)

    def _confusion(row) -> str:
        if row.sw_true == 1 and row.sw_pred == 1:
            return "TP"
        if row.sw_true == 0 and row.sw_pred == 1:
            return "FP"
        if row.sw_true == 1 and row.sw_pred == 0:
            return "FN"
        return "TN"

    df["confusion"] = df.apply(_confusion, axis=1)
    return df


def cstype_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of precision/recall/F1 broken down by cs_type_norm."""
    rows = []
    for ctype in df["cs_type_norm"].unique():
        sub = df[df["cs_type_norm"] == ctype]
        y_true = sub["sw_true"].values
        y_pred = sub["sw_pred"].values
        rows.append({
            "cs_type":   ctype,
            "n_tokens":  len(sub),
            "n_switches": int((y_true == 1).sum()),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall":    recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1":        f1_score(y_true, y_pred, average="binary", zero_division=0),
        })
    return pd.DataFrame(rows).sort_values("cs_type").reset_index(drop=True)


def print_qualitative_summary(df: pd.DataFrame, pair_name: str) -> None:
    """Print confusion counts and per-cstype metrics for one language pair."""
    print(f"\n{'='*70}\nQUALITATIVE ANALYSIS: {pair_name}\n{'='*70}")
    print("\nConfusion counts by cs_type:")
    pivot = (
        df.groupby(["cs_type_norm", "confusion"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["TP", "FP", "FN", "TN"], fill_value=0)
    )
    print(pivot.to_string())

    print("\nPer-type metrics:")
    tbl = cstype_metrics_table(df)
    print(tbl.to_string(index=False))


def print_example_cases(
    df:           pd.DataFrame,
    dataset:      CodeSwitchDatasetWithMeta,
    tokenizer,
    confusion_cat: str = "FN",
    n:            int  = 3,
) -> None:
    """Print n example tokens for a given confusion category (TP/FP/FN/TN)."""
    sub = df[df["confusion"] == confusion_cat].head(n)
    print(f"\n── {confusion_cat} examples ──")
    for _, row in sub.iterrows():
        item    = dataset[row["sample_idx"]]
        ids     = item["input_ids"].tolist()
        tokens  = tokenizer.convert_ids_to_tokens(ids)
        t       = int(row["token_pos"])
        ctx_lo  = max(0, t - 3)
        ctx_hi  = min(len(tokens) - 1, t + 3)
        context = tokens[ctx_lo: ctx_hi + 1]
        print(
            f"  [{row['cs_type_norm']}]  pos={t}  "
            f"sw_true={int(row['sw_true'])}  sw_pred={int(row['sw_pred'])}  "
            f"prob={row['sw_prob']:.3f}\n"
            f"  context: {context}"
        )


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_cstype_metrics(
    df:          pd.DataFrame,
    pair_name:   str,
    output_path: str,
) -> None:
    """Grouped bar chart of Precision / Recall / F1 per cs_type for one pair."""
    tbl    = cstype_metrics_table(df)
    ctypes = tbl["cs_type"].tolist()
    x      = np.arange(len(ctypes))
    w      = 0.25

    fig, ax = plt.subplots(figsize=(max(7, len(ctypes) * 2), 5))
    ax.bar(x - w, tbl["precision"].values, w, label="Precision", color="#4C72B0", alpha=0.85)
    ax.bar(x,     tbl["recall"].values,    w, label="Recall",    color="#DD8452", alpha=0.85)
    ax.bar(x + w, tbl["f1"].values,        w, label="F1",        color="#55A868", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ctypes, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"{pair_name}: Switch P/R/F1 by Code-Switch Type", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.show()


def plot_confidence_distributions(
    df:          pd.DataFrame,
    pair_name:   str,
    output_path: str,
) -> None:
    """Histograms of P(switch) for TP vs FP split by cs_type."""
    ctypes = df["cs_type_norm"].unique()
    n      = len(ctypes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ctype in zip(axes, ctypes):
        sub    = df[df["cs_type_norm"] == ctype]
        tp_sub = sub[sub["confusion"] == "TP"]["sw_prob"]
        fp_sub = sub[sub["confusion"] == "FP"]["sw_prob"]
        ax.hist(tp_sub, bins=20, alpha=0.6, label="TP", color="#55A868")
        ax.hist(fp_sub, bins=20, alpha=0.6, label="FP", color="#C44E52")
        ax.set_title(ctype, fontsize=9)
        ax.set_xlabel("P(switch)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Count")
    fig.suptitle(f"{pair_name}: Confidence Distribution (TP vs FP)", fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.show()


def plot_error_stacks(
    df:          pd.DataFrame,
    pair_name:   str,
    output_path: str,
) -> None:
    """Stacked bar of TP / FP / FN counts per cs_type (excludes TN for readability)."""
    ctypes = sorted(df["cs_type_norm"].unique())
    tp_counts = [len(df[(df["cs_type_norm"] == c) & (df["confusion"] == "TP")]) for c in ctypes]
    fp_counts = [len(df[(df["cs_type_norm"] == c) & (df["confusion"] == "FP")]) for c in ctypes]
    fn_counts = [len(df[(df["cs_type_norm"] == c) & (df["confusion"] == "FN")]) for c in ctypes]
    x = np.arange(len(ctypes))

    fig, ax = plt.subplots(figsize=(max(6, len(ctypes) * 1.5), 5))
    ax.bar(x, tp_counts, label="TP", color="#55A868", alpha=0.85)
    ax.bar(x, fp_counts, bottom=tp_counts, label="FP", color="#C44E52", alpha=0.85)
    ax.bar(x, fn_counts,
           bottom=[t + f for t, f in zip(tp_counts, fp_counts)],
           label="FN", color="#8172B2", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ctypes, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Token count")
    ax.set_title(f"{pair_name}: Error Distribution by Code-Switch Type", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.show()


def save_qualitative_json(
    df:          pd.DataFrame,
    pair_name:   str,
    output_path: str,
) -> None:
    """Save per-token prediction records for one pair as JSON."""
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if hasattr(v, "item"):
                r[k] = v.item()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"pair": pair_name, "records": records}, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {output_path}")


def run_qualitative_analysis(
    model:      nn.Module,
    all_stats:  dict,
    pairs:      list[str],
    tokenizer,
    device:     torch.device,
    output_dir: str = "results/qualitative",
    cfg:        QualitativeConfig | None = None,
) -> Dict[str, pd.DataFrame]:
    """Run the full qualitative analysis pipeline for each pair.

    Returns a dict mapping pair_name → qualitative DataFrame.
    """
    if cfg is None:
        cfg = QualitativeConfig()

    dfs: Dict[str, pd.DataFrame] = {}
    out = Path(output_dir)

    for pair_key in pairs:
        if pair_key not in all_stats or not all_stats[pair_key]["processed_samples"]:
            print(f"  ⚠ {pair_key} not in stats, skipping.")
            continue

        samples  = all_stats[pair_key]["processed_samples"]
        split_at = int(len(samples) * cfg.train_ratio)
        val_samp = samples[split_at:]

        dataset = CodeSwitchDatasetWithMeta(val_samp, tokenizer, cfg.max_len)
        if len(dataset) == 0:
            continue

        print(f"\n── Qualitative analysis: {pair_key}  (val={len(dataset)} samples)")
        records = evaluate_per_token(model, dataset, tokenizer, device)
        df      = build_qualitative_df(records)
        if df.empty:
            continue

        dfs[pair_key] = df
        pair_slug = pair_key.replace(" ", "_").replace("/", "-")

        print_qualitative_summary(df, pair_key)
        plot_cstype_metrics(df,             pair_key, str(out / f"{pair_slug}_cstype_metrics.png"))
        plot_confidence_distributions(df,   pair_key, str(out / f"{pair_slug}_confidence.png"))
        plot_error_stacks(df,               pair_key, str(out / f"{pair_slug}_error_stacks.png"))
        save_qualitative_json(df,           pair_key, str(out / f"{pair_slug}_qualitative.json"))

    return dfs
