from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_permeation.validation_model import train_and_validate_external_model


def build_parser() -> argparse.ArgumentParser:
    """Build the validation-ready training command-line parser."""
    parser = argparse.ArgumentParser(
        description="Train a cross-validated ensemble and evaluate an untouched external holdout."
    )
    parser.add_argument("--dataset", default="data/final/clean_trial4.csv")
    parser.add_argument("--target-col", default="logkpl")
    parser.add_argument(
        "--protocol",
        choices=("strict-grouped", "paper-reproduction"),
        default="strict-grouped",
        help="Strict unseen-SMILES validation or the paper's original 85/15 random split.",
    )
    parser.add_argument("--split-strategy", choices=("grouped", "random"), default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--fast", action="store_true", help="Use fewer estimators for a quick smoke test.")
    return parser


def main() -> None:
    """Train the validation model and generate all validation artifacts."""
    args = build_parser().parse_args()
    train_and_validate_external_model(
        dataset_csv_path=args.dataset,
        target_col=args.target_col,
        protocol=args.protocol,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        output_dir=args.output_dir,
        fast=args.fast,
    )


if __name__ == "__main__":
    main()
