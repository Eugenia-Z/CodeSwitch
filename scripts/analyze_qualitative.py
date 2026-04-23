#!/usr/bin/env python3
"""
Qualitative per-token error analysis grouped by code-switch type.

Evaluates a trained model on the validation set of each specified language pair
and produces per-cs_type precision/recall/F1 tables, confidence distribution plots,
and error-count stacked-bar charts.

Usage
-----
    # Analyse all default pairs
    python scripts/analyze_qualitative.py \
        --checkpoint checkpoints/best_xlmr.pt \
        --data data/preprocessed.pkl

    # Analyse specific pairs only
    python scripts/analyze_qualitative.py \
        --checkpoint checkpoints/best_xlmr.pt \
        --data data/preprocessed.pkl \
        --pairs Korean-English German-English Chinese-English

    # XGLM backbone
    python scripts/analyze_qualitative.py \
        --backbone xglm \
        --checkpoint checkpoints/best_xglm.pt \
        --data data/preprocessed.pkl
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeswitch.config import (
    BACKBONE_MODEL_DEFAULTS, ModelConfig, QualitativeConfig, parse_pair,
)
from codeswitch.model import build_model
from codeswitch.qualitative import run_qualitative_analysis
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    mc  = ModelConfig()
    qcfg = QualitativeConfig()

    p = argparse.ArgumentParser(
        description="Qualitative CS-type error analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Trained model checkpoint (.pt)")
    p.add_argument("--data",       default="data/preprocessed.pkl")
    p.add_argument("--backbone",   default=mc.backbone,
                   choices=list(BACKBONE_MODEL_DEFAULTS.keys()))
    p.add_argument("--model",      default=None,
                   help="HF model ID (overrides backbone default)")
    p.add_argument("--pairs",      nargs="+", metavar="LANG1-LANG2",
                   help="Pairs to analyse (default: QualitativeConfig.pairs_to_analyze)")
    p.add_argument("--max-len",    type=int,   default=qcfg.max_len)
    p.add_argument("--train-ratio", type=float, default=qcfg.train_ratio)
    p.add_argument("--output-dir", default="results/qualitative")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model or BACKBONE_MODEL_DEFAULTS[args.backbone]

    print(f"Device:    {device}")
    print(f"Backbone:  {args.backbone}  |  Model: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")

    print(f"\nLoading data: {args.data}")
    with open(args.data, "rb") as f:
        all_stats = pickle.load(f)

    tokenizer    = AutoTokenizer.from_pretrained(model_name)
    model_config = ModelConfig(backbone=args.backbone, model_name=model_name, max_len=args.max_len)
    model        = build_model(model_config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    qcfg = QualitativeConfig(max_len=args.max_len, train_ratio=args.train_ratio)
    pairs = [p.replace("-", " ", 1) if " " not in p else p for p in args.pairs] \
            if args.pairs else qcfg.pairs_to_analyze

    # Normalise "Korean-English" → "Korean-English" (the dict key format)
    if args.pairs:
        pairs = ["-".join(parse_pair(p)) for p in args.pairs]

    run_qualitative_analysis(
        model, all_stats, pairs, tokenizer, device,
        output_dir=args.output_dir,
        cfg=qcfg,
    )


if __name__ == "__main__":
    main()
