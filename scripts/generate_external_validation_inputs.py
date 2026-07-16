from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_permeation.validation_data import generate_validation_input_files


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for validation input generation."""
    parser = argparse.ArgumentParser(
        description="Generate train_data.csv and external_test_predictions.csv from project artifacts."
    )
    parser.add_argument("--dataset", default="data/final/clean_trial4.csv", help="Complete modeling dataset CSV.")
    parser.add_argument(
        "--predictions",
        default="reports/tables/improved_test_errors.csv",
        help="Saved external/test prediction artifact CSV.",
    )
    parser.add_argument("--target-col", default=None, help="Experimental target column in the complete dataset.")
    parser.add_argument("--actual-col", default=None, help="Actual response column in the prediction artifact.")
    parser.add_argument("--pred-col", default=None, help="Predicted response column in the prediction artifact.")
    parser.add_argument("--match-col", default=None, help="Shared identifier used to remove test rows from training.")
    parser.add_argument("--group-col", default=None, help="Grouping column used when reproducing the split.")
    parser.add_argument("--test-size", type=float, default=0.20, help="External/test fraction used by the model run.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state used by the model run.")
    parser.add_argument("--output-dir", default=".", help="Directory for the generated validation CSV files.")
    return parser


def main() -> None:
    """Generate the validation input files using command-line arguments."""
    args = build_parser().parse_args()
    generate_validation_input_files(
        dataset_csv_path=args.dataset,
        predictions_csv_path=args.predictions,
        target_col=args.target_col,
        actual_col=args.actual_col,
        pred_col=args.pred_col,
        match_col=args.match_col,
        group_col=args.group_col,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
