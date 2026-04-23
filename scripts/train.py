#!/usr/bin/env python3
"""
Train a code-switching predictor.

Backbone choices
----------------
  --backbone xlmr   XLM-RoBERTa with manual causal masking (default)
  --backbone xglm   XGLM (GPT-style, natively causal)

Usage
-----
    # Train with all defaults (XLM-R, 15 pairs, 8 epochs)
    python scripts/train.py

    # GPT backbone
    python scripts/train.py --backbone xglm

    # Custom model name (overrides backbone default)
    python scripts/train.py --backbone xglm --model facebook/xglm-1.7B

    # Quick smoke-test
    python scripts/train.py --epochs 2 --batch-size 16 --max-samples 500

    # Custom language pair split
    python scripts/train.py \\
        --train-pairs Chinese-English Hindi-English Italian-English \\
        --zeroshot-pairs Korean-English Russian-English \\
        --epochs 5

    # Save metrics JSON
    python scripts/train.py --results-json results/train_results.json
"""
from __future__ import annotations
import argparse
import gc
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeswitch.config import (
    BACKBONE_MODEL_DEFAULTS, TRAIN_PAIRS, ZEROSHOT_PAIRS,
    DataConfig, ModelConfig, TrainConfig, parse_pair,
)
from codeswitch.data import build_datasets, make_collate_fn
from codeswitch.evaluate import evaluate, evaluate_per_pair, print_sigma_summary
from codeswitch.model import build_model
from codeswitch.results_json import save_results_json
from codeswitch.trainer import compute_class_weights, set_seed, train_epoch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    tc = TrainConfig()
    mc = ModelConfig()
    dc = DataConfig()

    p = argparse.ArgumentParser(
        description="Train code-switching predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",           default="data/preprocessed.pkl",
                   help="Preprocessed pickle from scripts/preprocess.py")
    p.add_argument("--backbone",       default=mc.backbone,
                   choices=list(BACKBONE_MODEL_DEFAULTS.keys()),
                   help="Model backbone architecture")
    p.add_argument("--model",          default=None,
                   help="HF model ID (overrides backbone default)")
    p.add_argument("--train-pairs",    nargs="+", metavar="LANG1-LANG2",
                   help="Training language pairs (default: config.TRAIN_PAIRS)")
    p.add_argument("--zeroshot-pairs", nargs="+", metavar="LANG1-LANG2",
                   help="Zero-shot pairs (default: config.ZEROSHOT_PAIRS)")
    p.add_argument("--epochs",         type=int,   default=tc.num_epochs)
    p.add_argument("--batch-size",     type=int,   default=tc.batch_size)
    p.add_argument("--base-lr",        type=float, default=tc.base_lr)
    p.add_argument("--head-lr-mul",    type=float, default=tc.head_lr_multiplier,
                   help="Head LR = base_lr × this multiplier")
    p.add_argument("--lambda-dur",     type=float, default=tc.lambda_dur,
                   help="Weight for duration loss (0 = switch-only)")
    p.add_argument("--max-len",        type=int,   default=mc.max_len)
    p.add_argument("--dropout",        type=float, default=mc.dropout)
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze encoder weights, train only heads")
    p.add_argument("--train-ratio",    type=float, default=dc.train_ratio)
    p.add_argument("--seed",           type=int,   default=tc.seed)
    p.add_argument("--num-workers",    type=int,   default=tc.num_workers)
    p.add_argument("--checkpoint",     default=None,
                   help="Path to save best model checkpoint (default: checkpoints/best_<backbone>.pt)")
    p.add_argument("--results",        default=None,
                   help="Path to save final results + history (default: results/train_<backbone>.pkl)")
    p.add_argument("--results-json",   default=None, metavar="PATH",
                   help="Also save the same metrics payload as JSON (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_name = args.model or BACKBONE_MODEL_DEFAULTS[args.backbone]

    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        base_lr=args.base_lr,
        head_lr_multiplier=args.head_lr_mul,
        lambda_dur=args.lambda_dur,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model_config = ModelConfig(
        backbone=args.backbone,
        model_name=model_name,
        max_len=args.max_len,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
    )

    ckpt_path    = Path(args.checkpoint or f"checkpoints/best_{args.backbone}.pt")
    results_path = Path(args.results    or f"results/train_{args.backbone}.pkl")

    train_pairs    = [parse_pair(p) for p in args.train_pairs]    if args.train_pairs    else TRAIN_PAIRS
    zeroshot_pairs = [parse_pair(p) for p in args.zeroshot_pairs] if args.zeroshot_pairs else ZEROSHOT_PAIRS

    set_seed(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    print(f"  Backbone: {args.backbone}  |  Model: {model_name}")

    print(f"\nLoading preprocessed data: {args.data}")
    with open(args.data, "rb") as f:
        all_stats = pickle.load(f)
    print(f"  Loaded {len(all_stats)} language pairs")

    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collate   = make_collate_fn(tokenizer.pad_token_id)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_ds, val_ds = build_datasets(
        all_stats, train_pairs, tokenizer,
        train_ratio=args.train_ratio,
        max_len=model_config.max_len,
    )
    train_loader = DataLoader(
        train_ds, batch_size=train_config.batch_size,
        shuffle=True, collate_fn=collate, num_workers=train_config.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_config.batch_size,
        shuffle=False, collate_fn=collate, num_workers=train_config.num_workers,
    )

    print("\nComputing inverse-frequency class weights...")
    sw_w, dur_w = compute_class_weights(train_loader)
    sw_w  = sw_w.to(device)
    dur_w = dur_w.to(device)

    sw_criterion  = nn.CrossEntropyLoss(weight=sw_w,  ignore_index=-100)
    dur_criterion = nn.CrossEntropyLoss(weight=dur_w)

    model = build_model(model_config).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(),       "lr": train_config.base_lr},
            {"params": model.switch_head.parameters(),   "lr": train_config.head_lr},
            {"params": model.duration_head.parameters(), "lr": train_config.head_lr},
        ],
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config.num_epochs
    )

    print(f"\n  Backbone:    {args.backbone}  ({model_name})")
    print(f"  Mode:        {'FROZEN encoder' if model_config.freeze_encoder else 'FULL FINE-TUNING'}")
    print(f"  Encoder LR:  {train_config.base_lr}  |  Head LR: {train_config.head_lr}")
    print(f"  Train pairs: {len(train_pairs)}  |  Zero-shot: {len(zeroshot_pairs)}")
    print(f"  Batches:     {len(train_loader)} train  /  {len(val_loader)} val")

    history:    list[dict] = []
    best_sw_f1: float      = 0.0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_config.num_epochs + 1):
        losses   = train_epoch(model, train_loader, optimizer, device,
                               sw_criterion, dur_criterion, train_config)
        metrics  = evaluate(model, val_loader, device)
        pair_f1s = {
            k: v["switch_f1"]
            for k, v in evaluate_per_pair(
                model, all_stats, train_pairs, tokenizer, device,
                batch_size=train_config.batch_size,
                train_ratio=args.train_ratio,
                max_len=model_config.max_len,
                num_workers=train_config.num_workers,
            ).items()
        }
        scheduler.step()

        history.append({
            "epoch":       epoch,
            "loss_sw":     losses["loss_sw"],
            "loss_dur":    losses["loss_dur"],
            "switch_f1":   metrics["switch_f1"],
            "duration_f1": metrics["duration_f1"],
            "pair_f1s":    pair_f1s,
        })

        print(
            f"Epoch {epoch:02d} | "
            f"loss_sw={losses['loss_sw']:.4f}  loss_dur={losses['loss_dur']:.4f} | "
            f"switch_F1={metrics['switch_f1']:.4f}  duration_F1={metrics['duration_f1']:.4f}"
        )
        for k, v in pair_f1s.items():
            print(f"         {k:<25} switch_F1={v:.4f}")

        if metrics["switch_f1"] > best_sw_f1:
            best_sw_f1 = metrics["switch_f1"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best switch F1: {best_sw_f1:.4f} → {ckpt_path}")

    # Final evaluation on best checkpoint
    print(f"\nLoading best checkpoint from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    eval_kwargs = dict(
        batch_size=train_config.batch_size,
        train_ratio=args.train_ratio,
        max_len=model_config.max_len,
        num_workers=train_config.num_workers,
    )
    final_train = evaluate_per_pair(model, all_stats, train_pairs,    tokenizer, device, **eval_kwargs)
    final_zs    = evaluate_per_pair(model, all_stats, zeroshot_pairs, tokenizer, device, **eval_kwargs) \
                  if zeroshot_pairs else {}

    print_sigma_summary(final_train, final_zs)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backbone":         args.backbone,
        "model_name":       model_name,
        "history":          history,
        "train_results":    final_train,
        "zeroshot_results": final_zs,
        "best_switch_f1":   best_sw_f1,
    }
    with open(results_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n✓ Results saved to: {results_path}")

    if args.results_json:
        json_path = Path(args.results_json)
        save_results_json(json_path, payload)
        print(f"✓ Results (JSON) saved to: {json_path}")


if __name__ == "__main__":
    main()
