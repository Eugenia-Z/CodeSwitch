#!/usr/bin/env python3
"""
Preprocess SwitchLingua dataset: run LID + label generation, save pickle.

The output pickle stores word-level LID tags alongside the subword tokens so
the same file can be loaded with any backbone tokenizer.  If you intend to
train with a different backbone (e.g. xglm vs xlmr), you only need to run
preprocessing once; the DataLoader re-tokenises on-the-fly.

Usage
-----
    # Process all 15 language pairs (default backbone: xlmr)
    python scripts/preprocess.py

    # Use XGLM tokenizer during preprocessing (affects stored 'tokens' field)
    python scripts/preprocess.py --backbone xglm

    # Custom output / sample count
    python scripts/preprocess.py --output data/small.pkl --max-samples 1000

    # Only specific pairs
    python scripts/preprocess.py --pairs Chinese-English Hindi-English

    # HuggingFace token (alternatively set HF_TOKEN env var)
    HF_TOKEN=hf_xxx python scripts/preprocess.py
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeswitch.auth import hf_login
from codeswitch.config import (
    ALL_LANGUAGE_PAIRS, BACKBONE_MODEL_DEFAULTS, DataConfig, ModelConfig, parse_pair,
)
from codeswitch.data import analyze_language_pair
from codeswitch.lid import ProductionLID
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    cfg_data  = DataConfig()
    cfg_model = ModelConfig()

    p = argparse.ArgumentParser(
        description="Preprocess SwitchLingua → LID labels → pickle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output",      default="data/preprocessed.pkl",
                   help="Output pickle path")
    p.add_argument("--max-samples", type=int, default=cfg_data.max_samples_per_pair,
                   help="Max samples per language pair")
    p.add_argument("--pairs",       nargs="+", metavar="LANG1-LANG2",
                   help="Language pairs to process (default: all 15)")
    p.add_argument("--dataset",     default=cfg_data.dataset_name,
                   help="HuggingFace dataset name")
    p.add_argument("--backbone",    default=cfg_model.backbone,
                   choices=list(BACKBONE_MODEL_DEFAULTS.keys()),
                   help="Backbone tokenizer to use for subword alignment")
    p.add_argument("--model",       default=None,
                   help="HF model ID for tokenizer (overrides backbone default)")
    p.add_argument("--lid-model",   default=cfg_model.lid_model,
                   help="LID pipeline model name")
    p.add_argument("--hf-token",    default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    hf_login(args.hf_token)

    from datasets import load_dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"  Train split size: {len(dataset['train']):,}")

    pairs = [parse_pair(p) for p in args.pairs] if args.pairs else ALL_LANGUAGE_PAIRS
    print(f"\nLanguage pairs to process: {len(pairs)}")
    for lang1, lang2 in pairs:
        print(f"  {lang1}-{lang2}")

    model_name = args.model or BACKBONE_MODEL_DEFAULTS[args.backbone]
    print(f"\nLoading tokenizer: {model_name}  (backbone: {args.backbone})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1
    lid    = ProductionLID(model_name=args.lid_model, device=device)

    all_stats: dict = {}
    for lang1, lang2 in pairs:
        try:
            stats = analyze_language_pair(
                dataset["train"], lang1, lang2, lid, tokenizer,
                max_samples=args.max_samples,
            )
            if stats is not None:
                all_stats[f"{lang1}-{lang2}"] = stats
        except Exception as e:
            print(f"\n[ERROR] {lang1}-{lang2}: {e}")

    total_samples = sum(s["total_samples"] for s in all_stats.values())
    total_tokens  = sum(s["total_tokens"]  for s in all_stats.values())
    print(f"\n{'='*60}")
    print(f"✓ Done: {len(all_stats)} / {len(pairs)} pairs processed")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total tokens:  {total_tokens:,}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(all_stats, f)
    print(f"\n✓ Saved to: {output}")


if __name__ == "__main__":
    main()
