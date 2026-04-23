#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on specified language pairs.

Backbone is inferred from the results pickle when available, or can be
specified explicitly with --backbone.

Usage
-----
    # Evaluate XLM-R checkpoint on all train pairs (default)
    python scripts/evaluate.py --checkpoint checkpoints/best_xlmr.pt

    # Evaluate XGLM checkpoint
    python scripts/evaluate.py --backbone xglm --checkpoint checkpoints/best_xglm.pt

    # Custom language pair split
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best_xlmr.pt \\
        --train-pairs Chinese-English Hindi-English \\
        --zeroshot-pairs Korean-English Russian-English

    # Save results
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best_xlmr.pt \\
        --output results/eval_results.pkl \\
        --results-json results/eval_results.json
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeswitch.config import (
    BACKBONE_MODEL_DEFAULTS, TRAIN_PAIRS, ZEROSHOT_PAIRS,
    ModelConfig, TrainConfig, parse_pair,
)
from codeswitch.evaluate import evaluate_per_pair, print_sigma_summary
from codeswitch.model import build_model
from codeswitch.results_json import save_results_json
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    tc = TrainConfig()
    mc = ModelConfig()

    p = argparse.ArgumentParser(
        description="Evaluate code-switching checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",     required=True,
                   help="Path to saved model checkpoint (.pt)")
    p.add_argument("--data",           default="data/preprocessed.pkl",
                   help="Preprocessed pickle from scripts/preprocess.py")
    p.add_argument("--backbone",       default=mc.backbone,
                   choices=list(BACKBONE_MODEL_DEFAULTS.keys()),
                   help="Model backbone architecture")
    p.add_argument("--model",          default=None,
                   help="HF model ID (overrides backbone default)")
    p.add_argument("--train-pairs",    nargs="+", metavar="LANG1-LANG2",
                   help="Pairs to treat as 'train' in the σ table (default: config.TRAIN_PAIRS)")
    p.add_argument("--zeroshot-pairs", nargs="+", metavar="LANG1-LANG2",
                   help="Pairs to treat as zero-shot (default: config.ZEROSHOT_PAIRS)")
    p.add_argument("--max-len",        type=int,   default=mc.max_len)
    p.add_argument("--dropout",        type=float, default=mc.dropout)
    p.add_argument("--batch-size",     type=int,   default=tc.batch_size)
    p.add_argument("--train-ratio",    type=float, default=0.8,
                   help="Train/val split ratio used during preprocessing")
    p.add_argument("--num-workers",    type=int,   default=tc.num_workers)
    p.add_argument("--output",         default=None,
                   help="Save results dict to this pickle path (optional)")
    p.add_argument("--results-json",   default=None, metavar="PATH",
                   help="Save the same metrics as JSON (optional)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = args.model or BACKBONE_MODEL_DEFAULTS[args.backbone]
    print(f"Backbone: {args.backbone}  |  Model: {model_name}")

    train_pairs    = [parse_pair(p) for p in args.train_pairs]    if args.train_pairs    else TRAIN_PAIRS
    zeroshot_pairs = [parse_pair(p) for p in args.zeroshot_pairs] if args.zeroshot_pairs else ZEROSHOT_PAIRS

    print(f"\nLoading data: {args.data}")
    with open(args.data, "rb") as f:
        all_stats = pickle.load(f)
    print(f"  {len(all_stats)} language pairs available")

    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_config = ModelConfig(
        backbone=args.backbone,
        model_name=model_name,
        dropout=args.dropout,
        max_len=args.max_len,
    )
    model = build_model(model_config).to(device)
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    eval_kwargs = dict(
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        max_len=args.max_len,
        num_workers=args.num_workers,
    )
    train_results = evaluate_per_pair(model, all_stats, train_pairs,    tokenizer, device, **eval_kwargs)
    zs_results    = evaluate_per_pair(model, all_stats, zeroshot_pairs, tokenizer, device, **eval_kwargs) \
                    if zeroshot_pairs else {}

    print_sigma_summary(train_results, zs_results)

    payload = {
        "backbone":         args.backbone,
        "model_name":       model_name,
        "train_results":    train_results,
        "zeroshot_results": zs_results,
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(payload, f)
        print(f"\n✓ Results saved to: {out}")

    if args.results_json:
        json_path = Path(args.results_json)
        save_results_json(json_path, payload)
        print(f"✓ Results (JSON) saved to: {json_path}")


if __name__ == "__main__":
    main()
