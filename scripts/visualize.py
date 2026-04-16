#!/usr/bin/env python3
"""
Generate plots from saved training / evaluation results.

Usage
-----
    # Generate all plots from a training run
    python scripts/visualize.py --results results/train_results.pkl

    # Custom output directory
    python scripts/visualize.py --results results/train_results.pkl --output-dir figures/

    # Only universality chart (no training history)
    python scripts/visualize.py --results results/eval_results.pkl --no-history
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeswitch.visualize import plot_training_history, plot_universality


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize training / evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results",    required=True,
                   help="Pickle file produced by scripts/train.py or scripts/evaluate.py")
    p.add_argument("--output-dir", default="results/",
                   help="Directory for saved PNG files")
    p.add_argument("--no-history", action="store_true",
                   help="Skip training history plot (useful for eval-only pickles)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.results}")
    with open(args.results, "rb") as f:
        data = pickle.load(f)

    train_results    = data.get("train_results",    {})
    zeroshot_results = data.get("zeroshot_results", {})
    history          = data.get("history",          [])

    if train_results or zeroshot_results:
        plot_universality(
            train_results, zeroshot_results,
            output_path=str(outdir / "xlmr_universality.png"),
        )
    else:
        print("No per-pair results found in pickle — skipping universality chart.")

    if history and not args.no_history:
        plot_training_history(
            history,
            output_path=str(outdir / "training_history.png"),
        )
    elif not history:
        print("No training history found in pickle — skipping history plot.")


if __name__ == "__main__":
    main()
