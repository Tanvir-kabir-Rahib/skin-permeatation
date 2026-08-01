from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_permeation.benchmark_external_validation import validate_all_benchmark_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrain and externally validate every saved benchmark model on one common holdout."
    )
    parser.add_argument("--dataset", default="data/final/clean_trial4.csv")
    parser.add_argument("--target-col", default="logkpl")
    parser.add_argument(
        "--protocol",
        choices=("strict-grouped", "paper-reproduction"),
        default="paper-reproduction",
    )
    parser.add_argument("--split-strategy", choices=("grouped", "random"), default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--benchmark-metrics", default="reports/tables/benchmark_metrics.csv")
    parser.add_argument("--models-dir", default="models/reproduction/benchmark")
    parser.add_argument("--output-dir", default="outputs/external_validation/benchmark_models")
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate_all_benchmark_models(
        dataset_csv_path=args.dataset,
        target_col=args.target_col,
        protocol=args.protocol,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
        benchmark_metrics_path=args.benchmark_metrics,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        save_fitted_models=not args.no_save_models,
        make_plots=not args.no_plots,
    )
    print(result.summary[["Model", "R2_ext", "RMSE_ext", "CCC_ext", "Criteria Passed", "All Criteria Pass"]].to_string(index=False))
    print(f"\nOutputs saved to: {result.output_dir.resolve()}")
    if result.protocol["split"]["overlapping_validation_rows"]:
        print(
            "Caveat: this paper-reproduction split includes "
            f"{result.protocol['split']['overlapping_validation_rows']} validation rows whose SMILES also occur in training."
        )


if __name__ == "__main__":
    main()
