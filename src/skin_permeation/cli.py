from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .logging_utils import configure_logging
from .paths import ProjectPaths

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Skin permeability paper reproduction toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit-data")
    subparsers.add_parser("validate-descriptors")
    subparsers.add_parser("reproducibility-report")

    baseline = subparsers.add_parser("run-baseline")
    baseline.add_argument("--config", default="configs/paper_baseline.yaml")

    improved = subparsers.add_parser("run-improved")
    improved.add_argument("--config", default="configs/improved.yaml")

    benchmark = subparsers.add_parser("run-benchmark")
    benchmark.add_argument("--config", default="configs/benchmark.yaml")

    cluster = subparsers.add_parser("cluster-drugbank")
    cluster.add_argument("--model-path", default="models/reproduction/benchmark/best_benchmark_model.joblib")
    cluster.add_argument("--scaler-path", default="")

    atc = subparsers.add_parser("run-atc")
    atc.add_argument("--predictions", default="reports/tables/drugbank_predictions_and_clusters.csv")

    java = subparsers.add_parser("run-java-descriptors")
    java.add_argument("--java", default="java")
    java.add_argument("--maven", default="mvn")

    predict = subparsers.add_parser("predict-logkp")
    predict.add_argument("--smiles", default="")
    predict.add_argument("--texpi", type=float, default=None)
    predict.add_argument("--plot-output", default="figures/predictions/predicted_vs_actual_logkp.png")
    predict.add_argument("--show-plot", action="store_true")
    predict.add_argument("--rdkit-python", default=None)
    predict.add_argument("--maven", default="mvn")

    full = subparsers.add_parser("full-run")
    full.add_argument("--baseline-config", default="configs/paper_baseline.yaml")
    full.add_argument("--benchmark-config", default="configs/benchmark.yaml")
    full.add_argument("--improved-config", default="configs/improved.yaml")
    full.add_argument("--model-path", default="models/reproduction/benchmark/best_benchmark_model.joblib")
    full.add_argument("--scaler-path", default="")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.discover()
    paths.ensure_runtime_dirs()
    configure_logging(paths.reports / "logs" / "pipeline.log")

    if args.command == "audit-data":
        from .data_audit import run_audit

        run_audit(paths)
        return
    if args.command == "validate-descriptors":
        from .descriptors import validate_descriptor_inventory

        validate_descriptor_inventory(paths)
        return
    if args.command == "reproducibility-report":
        from .paper_reconstruction import write_reproducibility_report

        write_reproducibility_report(paths)
        return
    if args.command == "run-baseline":
        from .modeling.baseline import run_baseline

        run_baseline(paths, Path(args.config))
        return
    if args.command == "run-improved":
        from .modeling.improved import run_improved

        run_improved(paths, Path(args.config))
        return
    if args.command == "run-benchmark":
        from .modeling.benchmark import run_benchmark

        run_benchmark(paths, Path(args.config))
        return
    if args.command == "cluster-drugbank":
        from .analysis.clustering import run_clustering

        scaler_path = Path(args.scaler_path) if args.scaler_path else None
        run_clustering(paths, Path(args.model_path), scaler_path)
        return
    if args.command == "run-atc":
        import pandas as pd
        from .analysis.atc import run_atc_analysis

        predicted = pd.read_csv(args.predictions)
        run_atc_analysis(paths, predicted)
        return
    if args.command == "run-java-descriptors":
        from .descriptors import run_java_descriptor_generator

        run_java_descriptor_generator(paths, java_executable=args.java, maven_executable=args.maven)
        return
    if args.command == "predict-logkp":
        from .prediction import run_prediction_command

        run_prediction_command(args, paths)
        return
    if args.command == "full-run":
        from .analysis.atc import run_atc_analysis
        from .analysis.clustering import run_clustering
        from .data_audit import run_audit
        from .descriptors import validate_descriptor_inventory
        from .modeling.baseline import run_baseline
        from .modeling.benchmark import run_benchmark
        from .modeling.improved import run_improved
        from .paper_reconstruction import write_reproducibility_report
        from .reporting import build_markdown_report

        run_audit(paths)
        validate_descriptor_inventory(paths)
        write_reproducibility_report(paths)
        run_baseline(paths, Path(args.baseline_config))
        run_benchmark(paths, Path(args.benchmark_config))
        run_improved(paths, Path(args.improved_config))
        scaler_path = Path(args.scaler_path) if args.scaler_path else None
        clustered = run_clustering(paths, Path(args.model_path), scaler_path)
        try:
            run_atc_analysis(paths, clustered)
        except FileNotFoundError as exc:
            LOGGER.warning("ATC analysis skipped: %s", exc)
        (paths.root / "REPORT.md").write_text(build_markdown_report(paths), encoding="utf-8")
        return


if __name__ == "__main__":
    main()
