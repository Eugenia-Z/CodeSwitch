#!/usr/bin/env python3
"""
Burstiness analysis: compare multitask (λ=1.0) vs single-task (λ=0.0) switch recall
in bursty vs isolated code-switching regions.

Usage
-----
    # Analyze multitask checkpoint only (no single-task comparison)
    python scripts/analyze_burstiness.py \
        --checkpoint checkpoints/best_xlmr.pt \
        --data data/preprocessed.pkl

    # Full comparison (provide both checkpoints)
    python scripts/analyze_burstiness.py \
        --checkpoint checkpoints/best_xlmr.pt \
        --st-checkpoint checkpoints/best_xlmr_st.pt \
        --data data/preprocessed.pkl \
        --output-dir results/burstiness

    # Use XGLM backbone
    python scripts/analyze_burstiness.py \
        --backbone xglm \
        --checkpoint checkpoints/best_xglm.pt \
        --data data/preprocessed.pkl
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeswitch.burstiness import (
    compute_burstiness_metrics,
    plot_burst_delta,
    plot_burstiness_comparison,
    plot_density_recall_scatter,
    print_burstiness_table,
)
from codeswitch.config import (
    BACKBONE_MODEL_DEFAULTS, TRAIN_PAIRS, BurstinessConfig, ModelConfig, parse_pair,
)
from codeswitch.model import build_model
from codeswitch.results_json import save_results_json
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    mc  = ModelConfig()
    cfg = BurstinessConfig()

    p = argparse.ArgumentParser(
        description="Burstiness analysis: multitask vs single-task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",    required=True,
                   help="Multitask checkpoint (λ=1.0)")
    p.add_argument("--st-checkpoint", default=None,
                   help="Single-task checkpoint (λ=0.0); enables MT vs ST comparison")
    p.add_argument("--data",          default="data/preprocessed.pkl")
    p.add_argument("--backbone",      default=mc.backbone,
                   choices=list(BACKBONE_MODEL_DEFAULTS.keys()))
    p.add_argument("--model",         default=None,
                   help="HF model ID (overrides backbone default)")
    p.add_argument("--pairs",         nargs="+", metavar="LANG1-LANG2",
                   help="Pairs to analyse (default: all TRAIN_PAIRS)")
    p.add_argument("--burst-threshold", type=int,   default=cfg.burst_threshold)
    p.add_argument("--window-size",     type=int,   default=cfg.window_size)
    p.add_argument("--max-len",         type=int,   default=cfg.max_len)
    p.add_argument("--train-ratio",     type=float, default=cfg.train_ratio)
    p.add_argument("--output-dir",      default="results/burstiness")
    p.add_argument("--results-json",    default=None, metavar="PATH")
    return p.parse_args()


def _load_model(backbone: str, model_name: str, checkpoint: str, device: torch.device):
    cfg   = ModelConfig(backbone=backbone, model_name=model_name)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def main() -> None:
    args       = parse_args()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model or BACKBONE_MODEL_DEFAULTS[args.backbone]

    print(f"Device:    {device}")
    print(f"Backbone:  {args.backbone}  |  Model: {model_name}")
    print(f"Checkpoint (MT): {args.checkpoint}")
    if args.st_checkpoint:
        print(f"Checkpoint (ST): {args.st_checkpoint}")

    print(f"\nLoading data: {args.data}")
    with open(args.data, "rb") as f:
        all_stats = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pairs = [parse_pair(p) for p in args.pairs] if args.pairs else TRAIN_PAIRS

    burst_cfg = BurstinessConfig(
        burst_threshold=args.burst_threshold,
        window_size=args.window_size,
        max_len=args.max_len,
        train_ratio=args.train_ratio,
    )

    print("\nLoading multitask model...")
    mt_model = _load_model(args.backbone, model_name, args.checkpoint, device)

    print("\nComputing multitask burstiness metrics...")
    mt_results = compute_burstiness_metrics(mt_model, all_stats, pairs, tokenizer, device, burst_cfg)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.st_checkpoint:
        print("\nLoading single-task model...")
        st_model = _load_model(args.backbone, model_name, args.st_checkpoint, device)
        print("\nComputing single-task burstiness metrics...")
        st_results = compute_burstiness_metrics(st_model, all_stats, pairs, tokenizer, device, burst_cfg)

        print_burstiness_table(mt_results, st_results)
        plot_burstiness_comparison(mt_results, st_results, str(out / "burstiness_comparison.png"))
        plot_burst_delta(mt_results, st_results, str(out / "burstiness_delta.png"))
    else:
        print_burstiness_table(mt_results)

    plot_density_recall_scatter(mt_results, all_stats, str(out / "burstiness_density_scatter.png"))

    payload: dict = {"mt": mt_results}
    if args.st_checkpoint:
        payload["st"] = st_results

    if args.results_json:
        save_results_json(args.results_json, payload)
        print(f"✓ JSON saved: {args.results_json}")
    else:
        save_results_json(str(out / "burstiness_results.json"), payload)
        print(f"✓ JSON saved: {out / 'burstiness_results.json'}")


if __name__ == "__main__":
    main()
