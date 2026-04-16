#!/usr/bin/env python3
"""
Evaluate a saved XLM-R checkpoint on specified language pairs.

Usage
-----
    # Evaluate on all train pairs (default)
    python scripts/evaluate.py --checkpoint checkpoints/best_xlmr.pt

    # Specify pairs manually
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best_xlmr.pt \\
        --train-pairs Chinese-English Hindi-English \\
        --zeroshot-pairs Korean-English Russian-English

    # Save results to pickle for later visualisation
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best_xlmr.pt \\
        --output results/eval_results.pkl
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeswitch.config import TRAIN_PAIRS, ZEROSHOT_PAIRS, ModelConfig, TrainConfig, parse_pair
from codeswitch.evaluate import evaluate_per_pair, print_sigma_summary
from codeswitch.model import XLMRCodeSwitchPredictor
from transformers import XLMRobertaTokenizer


def parse_args() -> argparse.Namespace:
    tc = TrainConfig()
    mc = ModelConfig()

    p = argparse.ArgumentParser(
        description="Evaluate XLM-R checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",     required=True,
                   help="Path to saved model checkpoint (.pt)")
    p.add_argument("--data",           default="data/preprocessed.pkl",
                   help="Preprocessed pickle from scripts/preprocess.py")
    p.add_argument("--train-pairs",    nargs="+", metavar="LANG1-LANG2",
                   help="Pairs to treat as 'train' in the σ table (default: config.TRAIN_PAIRS)")
    p.add_argument("--zeroshot-pairs", nargs="+", metavar="LANG1-LANG2",
                   help="Pairs to treat as zero-shot (default: config.ZEROSHOT_PAIRS)")
    p.add_argument("--model",          default=mc.model_name)
    p.add_argument("--max-len",        type=int,   default=mc.max_len)
    p.add_argument("--dropout",        type=float, default=mc.dropout)
    p.add_argument("--batch-size",     type=int,   default=tc.batch_size)
    p.add_argument("--train-ratio",    type=float, default=0.8,
                   help="Train/val split ratio used during preprocessing")
    p.add_argument("--num-workers",    type=int,   default=tc.num_workers)
    p.add_argument("--output",         default=None,
                   help="Save results dict to this pickle path (optional)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_pairs    = [parse_pair(p) for p in args.train_pairs]    if args.train_pairs    else TRAIN_PAIRS
    zeroshot_pairs = [parse_pair(p) for p in args.zeroshot_pairs] if args.zeroshot_pairs else ZEROSHOT_PAIRS

    print(f"\nLoading data: {args.data}")
    with open(args.data, "rb") as f:
        all_stats = pickle.load(f)
    print(f"  {len(all_stats)} language pairs available")

    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model)

    model = XLMRCodeSwitchPredictor(
        model_name=args.model, dropout=args.dropout
    ).to(device)
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

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump({"train_results": train_results, "zeroshot_results": zs_results}, f)
        print(f"\n✓ Results saved to: {out}")


if __name__ == "__main__":
    main()
