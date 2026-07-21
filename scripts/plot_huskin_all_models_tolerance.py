"""Create tolerance-filtered HuSkinDB outputs for every saved model.

The experimental target is used to select rows, so all subset metrics and
figures are descriptive only and must not be treated as unbiased validation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_huskin_tolerance import (
    BLUE,
    GOLD,
    GRID,
    INK,
    calculate_subset_metrics,
    configure_plot_style,
    infer_model_slug,
    run,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT = Path("results/huskinDB/huskin_predictions.csv")
DEFAULT_METRICS = Path("results/huskinDB/huskin_model_metrics.csv")
DEFAULT_OUTPUT = Path("results/huskinDB/within_0.6_all_models")
DEFAULT_EXPERIMENTAL_COLUMN = "logkp (cm/s)"
DEFAULT_TOLERANCE = 0.6


def load_model_specs(metrics_path: Path) -> pd.DataFrame:
    """Load the model display names and prediction columns in metric order."""

    metrics = pd.read_csv(metrics_path)
    required = {"Model", "Prediction_Column", "RMSE", "R2"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Missing required columns in {metrics_path}: {missing}")
    duplicated = metrics["Prediction_Column"].duplicated(keep=False)
    if duplicated.any():
        duplicate_names = metrics.loc[duplicated, "Prediction_Column"].tolist()
        raise ValueError(f"Duplicate prediction columns in {metrics_path}: {duplicate_names}")
    return metrics.reset_index(drop=True)


def build_summary_row(
    full_frame: pd.DataFrame,
    filtered: pd.DataFrame,
    model_name: str,
    prediction_column: str,
    model_slug: str,
    tolerance: float,
    overall_rank: int,
    overall_rmse: float,
    overall_r2: float,
    csv_path: Path,
    plot_paths: list[Path],
) -> dict[str, object]:
    """Build one audited summary row for a processed model."""

    filtered_prediction_column = f"Predicted_logKp_{model_slug}"
    absolute_error_column = f"Absolute_Error_{model_slug}"
    metrics = calculate_subset_metrics(filtered, filtered_prediction_column)
    experimental = pd.to_numeric(
        full_frame[DEFAULT_EXPERIMENTAL_COLUMN], errors="coerce"
    )
    predicted = pd.to_numeric(full_frame[prediction_column], errors="coerce")
    valid_pairs = int((experimental.notna() & predicted.notna()).sum())
    selected_rows = len(filtered)
    return {
        "Overall_Rank": overall_rank,
        "Model": model_name,
        "Model_Slug": model_slug,
        "Prediction_Column": prediction_column,
        "Tolerance": tolerance,
        "Valid_Pairs": valid_pairs,
        "Selected_Rows": selected_rows,
        "Selected_Row_Percent": (
            100.0 * selected_rows / valid_pairs if valid_pairs else float("nan")
        ),
        "Unique_Canonical_Compounds": int(
            filtered["Canonical_SMILES"].nunique(dropna=True)
        ),
        "Subset_R2": metrics["r2"],
        "Subset_RMSE": metrics["rmse"],
        "Subset_MAE": metrics["mae"],
        "Subset_Bias": metrics["bias"],
        "Maximum_Absolute_Error": float(filtered[absolute_error_column].max()),
        "Overall_RMSE": overall_rmse,
        "Overall_R2": overall_r2,
        "Filtered_CSV": str(csv_path),
        "Experimental_vs_Predicted_Plot": str(plot_paths[0]),
        "Residual_Plot": str(plot_paths[1]),
        "Absolute_Error_Plot": str(plot_paths[2]),
    }


def plot_model_comparison(
    summary: pd.DataFrame,
    output_path: Path,
    tolerance: float,
) -> None:
    """Plot selected-row coverage and subset errors for all models."""

    ordered = summary.sort_values(
        ["Selected_Rows", "Overall_Rank"], ascending=[True, False]
    ).reset_index(drop=True)
    labels = ordered["Model"].astype(str).tolist()
    positions = np.arange(len(ordered))

    configure_plot_style()
    fig, (coverage_ax, error_ax) = plt.subplots(
        1,
        2,
        figsize=(12.0, 7.2),
        gridspec_kw={"width_ratios": [1.0, 1.12]},
    )
    coverage_ax.barh(
        positions,
        ordered["Selected_Rows"],
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )
    coverage_ax.set(
        title="Measurements retained",
        xlabel="Selected measurement rows",
        yticks=positions,
        yticklabels=labels,
        xlim=(0, max(ordered["Valid_Pairs"].max() * 1.08, 1)),
    )
    coverage_ax.grid(True, axis="x")
    for index, row in ordered.iterrows():
        coverage_ax.text(
            row["Selected_Rows"] + ordered["Valid_Pairs"].max() * 0.012,
            index,
            f"{int(row['Selected_Rows'])} ({row['Selected_Row_Percent']:.1f}%)",
            va="center",
            fontsize=8,
            color=INK,
        )

    bar_height = 0.36
    error_ax.barh(
        positions - bar_height / 2,
        ordered["Subset_RMSE"],
        height=bar_height,
        color=BLUE,
        label="Subset RMSE",
        alpha=0.9,
    )
    error_ax.barh(
        positions + bar_height / 2,
        ordered["Subset_MAE"],
        height=bar_height,
        color=GOLD,
        label="Subset MAE",
        alpha=0.9,
    )
    error_ax.set(
        title="Errors within selected subsets",
        xlabel="Error in logKp (cm/s)",
        yticks=positions,
        yticklabels=[],
        xlim=(0, max(ordered["Subset_RMSE"].max() * 1.15, 0.01)),
    )
    error_ax.grid(True, axis="x")
    error_ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    fig.suptitle(
        f"HuSkinDB model comparison within ±{tolerance:.2f}",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Rows were selected using experimental logKp; subset metrics are "
            "descriptive and not unbiased model-performance estimates."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.045, 0.9, 0.95), w_pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_all_models(
    input_path: Path,
    metrics_path: Path,
    output_dir: Path,
    experimental_column: str,
    tolerance: float,
) -> tuple[Path, Path, pd.DataFrame]:
    """Generate tolerance-filtered CSVs and plots for all metric-table models."""

    if tolerance < 0:
        raise ValueError("Tolerance must be non-negative.")
    full_frame = pd.read_csv(input_path)
    if experimental_column != DEFAULT_EXPERIMENTAL_COLUMN:
        full_frame = full_frame.rename(
            columns={experimental_column: DEFAULT_EXPERIMENTAL_COLUMN}
        )
    if DEFAULT_EXPERIMENTAL_COLUMN not in full_frame.columns:
        raise ValueError(
            f"Missing experimental column {experimental_column!r} in {input_path}."
        )

    model_specs = load_model_specs(metrics_path)
    missing_predictions = sorted(
        set(model_specs["Prediction_Column"]).difference(full_frame.columns)
    )
    if missing_predictions:
        raise ValueError(
            f"Prediction columns listed in {metrics_path} are missing from "
            f"{input_path}: {missing_predictions}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for zero_based_rank, spec in model_specs.iterrows():
        model_name = str(spec["Model"])
        prediction_column = str(spec["Prediction_Column"])
        model_slug = infer_model_slug(prediction_column)
        model_output_dir = output_dir / model_slug
        csv_path, plot_paths = run(
            input_path=input_path,
            output_dir=model_output_dir,
            experimental_column=experimental_column,
            prediction_column=prediction_column,
            tolerance=tolerance,
            model_name=model_name,
            model_slug=model_slug,
        )
        filtered = pd.read_csv(csv_path)
        summary_rows.append(
            build_summary_row(
                full_frame=full_frame,
                filtered=filtered,
                model_name=model_name,
                prediction_column=prediction_column,
                model_slug=model_slug,
                tolerance=tolerance,
                overall_rank=zero_based_rank + 1,
                overall_rmse=float(spec["RMSE"]),
                overall_r2=float(spec["R2"]),
                csv_path=csv_path,
                plot_paths=plot_paths,
            )
        )

    summary = pd.DataFrame(summary_rows).sort_values("Overall_Rank")
    tolerance_label = f"{tolerance:.4f}".rstrip("0").rstrip(".")
    summary_path = (
        output_dir / f"huskin_all_models_within_{tolerance_label}_summary.csv"
    )
    comparison_path = (
        output_dir / f"all_models_comparison_within_{tolerance_label}.png"
    )
    summary.to_csv(summary_path, index=False, float_format="%.8f")
    plot_model_comparison(summary, comparison_path, tolerance)
    LOGGER.info(
        "Processed %d model entries; summary saved to %s.",
        len(summary),
        summary_path,
    )
    return summary_path, comparison_path, summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Create tolerance-filtered HuSkinDB outputs for all models."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experimental-column", default=DEFAULT_EXPERIMENTAL_COLUMN)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser


def main() -> int:
    """Run the all-model command-line interface."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    summary_path, comparison_path, _ = run_all_models(
        input_path=args.input,
        metrics_path=args.metrics,
        output_dir=args.output,
        experimental_column=args.experimental_column,
        tolerance=args.tolerance,
    )
    LOGGER.info("Summary: %s", summary_path)
    LOGGER.info("Comparison plot: %s", comparison_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
