"""Filter HuSkinDB predictions by absolute error and create diagnostic plots.

This is a post-prediction reporting utility.  The filtered subset must not be
used as an unbiased estimate of model performance because the experimental
target is used to select the displayed rows.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT = Path("results/huskinDB/huskin_predictions.csv")
DEFAULT_OUTPUT = Path("results/huskinDB/within_0.05_LGBM")
DEFAULT_EXPERIMENTAL_COLUMN = "logkp (cm/s)"
DEFAULT_PREDICTION_COLUMN = "Predicted_logKp_LGBM"
DEFAULT_MODEL_NAME = "LGBM"
DEFAULT_TOLERANCE = 0.05

BLUE = "#2563EB"
GOLD = "#D4A72C"
INK = "#1F2937"
GRID = "#D1D5DB"
PALE_BLUE = "#DBEAFE"


def normalize_model_slug(value: str) -> str:
    """Return a filesystem- and column-safe identifier for a model."""

    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    if not slug:
        raise ValueError("Model name must contain at least one letter or number.")
    return slug


def infer_model_slug(prediction_column: str, model_slug: str | None = None) -> str:
    """Infer a stable model identifier from a prediction-column name."""

    if model_slug:
        return normalize_model_slug(model_slug)
    prefix = "Predicted_logKp_"
    raw_slug = (
        prediction_column[len(prefix) :]
        if prediction_column.startswith(prefix)
        else prediction_column
    )
    return normalize_model_slug(raw_slug)


def configure_plot_style() -> None:
    """Apply a consistent publication-oriented Matplotlib style."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "font.size": 10,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def load_and_filter_predictions(
    input_path: Path,
    experimental_column: str,
    prediction_column: str,
    tolerance: float,
    model_slug: str = "LGBM",
) -> pd.DataFrame:
    """Return rows whose prediction is within ``tolerance`` of experiment."""

    if tolerance < 0:
        raise ValueError("Tolerance must be non-negative.")

    frame = pd.read_csv(input_path)
    required = {
        "Compound name",
        "Smiles",
        experimental_column,
        prediction_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    experimental = pd.to_numeric(frame[experimental_column], errors="coerce")
    predicted = pd.to_numeric(frame[prediction_column], errors="coerce")
    prediction_error = predicted - experimental
    absolute_error = prediction_error.abs()
    valid = experimental.notna() & predicted.notna()
    selected = valid & absolute_error.le(tolerance + np.finfo(float).eps)

    canonical = (
        frame["Canonical_SMILES"]
        if "Canonical_SMILES" in frame.columns
        else frame["Smiles"]
    )
    normalized_slug = normalize_model_slug(model_slug)
    filtered_prediction_column = f"Predicted_logKp_{normalized_slug}"
    prediction_error_column = f"Prediction_Error_{normalized_slug}"
    absolute_error_column = f"Absolute_Error_{normalized_slug}"
    filtered = pd.DataFrame(
        {
            "Source_Data_Row": np.arange(1, len(frame) + 1, dtype=int),
            "Compound_Name": frame["Compound name"],
            "Smiles": frame["Smiles"],
            "Canonical_SMILES": canonical,
            "Experimental_logKp_cm_s": experimental,
            filtered_prediction_column: predicted,
            prediction_error_column: prediction_error,
            absolute_error_column: absolute_error,
        }
    ).loc[selected]

    return filtered.reset_index(drop=True)


def calculate_subset_metrics(
    frame: pd.DataFrame,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
) -> dict[str, float]:
    """Calculate descriptive metrics for the selected rows."""

    experimental = frame["Experimental_logKp_cm_s"].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=float)
    errors = predicted - experimental
    denominator = np.sum((experimental - experimental.mean()) ** 2)
    r2 = float("nan") if denominator == 0 else float(1 - np.sum(errors**2) / denominator)
    return {
        "r2": r2,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
    }


def add_selection_note(fig: plt.Figure, tolerance: float) -> None:
    """Add the essential selection-bias caveat to a figure."""

    fig.text(
        0.5,
        0.012,
        (
            f"Post-prediction subset selected using |predicted − experimental| ≤ {tolerance:.2f}; "
            "not an unbiased model-performance estimate."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4B5563",
    )


def plot_experimental_vs_predicted(
    frame: pd.DataFrame,
    output_path: Path,
    tolerance: float,
    model_name: str = DEFAULT_MODEL_NAME,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
) -> None:
    """Create an experimental-versus-predicted scatter plot."""

    experimental = frame["Experimental_logKp_cm_s"].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=float)
    metrics = calculate_subset_metrics(frame, prediction_column)

    lower = float(min(experimental.min(), predicted.min()) - 0.2)
    upper = float(max(experimental.max(), predicted.max()) + 0.2)
    identity = np.linspace(lower, upper, 300)

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.fill_between(
        identity,
        identity - tolerance,
        identity + tolerance,
        color=PALE_BLUE,
        alpha=0.85,
        label=f"±{tolerance:.2f} selection band",
        zorder=1,
    )
    ax.plot(identity, identity, color=INK, linestyle="--", linewidth=1.4, label="Identity line", zorder=2)
    ax.scatter(
        experimental,
        predicted,
        s=42,
        color=BLUE,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.88,
        label="Selected measurements",
        zorder=3,
    )
    ax.set(
        title=f"HuSkinDB {model_name} predictions within ±{tolerance:.2f}",
        xlabel="Experimental logKp (cm/s)",
        ylabel="Predicted logKp (cm/s)",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.text(
        0.98,
        0.04,
        (
            f"n = {len(frame)} rows\n"
            f"R² = {metrics['r2']:.4f}\n"
            f"RMSE = {metrics['rmse']:.4f}\n"
            f"MAE = {metrics['mae']:.4f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": GRID, "alpha": 0.95},
    )
    add_selection_note(fig, tolerance)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    frame: pd.DataFrame,
    output_path: Path,
    tolerance: float,
    model_name: str = DEFAULT_MODEL_NAME,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
) -> None:
    """Create a residual-versus-prediction plot for the selected rows."""

    predicted = frame[prediction_column].to_numpy(dtype=float)
    residual = (
        frame["Experimental_logKp_cm_s"].to_numpy(dtype=float) - predicted
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.axhspan(-tolerance, tolerance, color=PALE_BLUE, alpha=0.8, label=f"±{tolerance:.2f} band")
    ax.axhline(0, color=INK, linestyle="--", linewidth=1.4)
    ax.axhline(tolerance, color=BLUE, linestyle=":", linewidth=1.0)
    ax.axhline(-tolerance, color=BLUE, linestyle=":", linewidth=1.0)
    ax.scatter(
        predicted,
        residual,
        s=40,
        color=BLUE,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.88,
    )
    ax.set(
        title=f"Residuals for {model_name} matches within ±{tolerance:.2f}",
        xlabel="Predicted logKp (cm/s)",
        ylabel="Residual (experimental − predicted)",
        ylim=(-tolerance * 1.35, tolerance * 1.35),
    )
    ax.grid(True)
    ax.legend(loc="upper right")
    add_selection_note(fig, tolerance)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_error_distribution(
    frame: pd.DataFrame,
    output_path: Path,
    tolerance: float,
    model_name: str = DEFAULT_MODEL_NAME,
    absolute_error_column: str = "Absolute_Error_LGBM",
) -> None:
    """Create a histogram of absolute errors within the selected subset."""

    absolute_error = frame[absolute_error_column].to_numpy(dtype=float)
    bins = np.linspace(0, tolerance, 11)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    counts, _, _ = ax.hist(
        absolute_error,
        bins=bins,
        color=GOLD,
        edgecolor="white",
        linewidth=1.0,
        alpha=0.92,
    )
    ax.axvline(
        absolute_error.mean(),
        color=INK,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {absolute_error.mean():.4f}",
    )
    ax.axvline(
        np.median(absolute_error),
        color=BLUE,
        linestyle=":",
        linewidth=1.8,
        label=f"Median = {np.median(absolute_error):.4f}",
    )
    ax.set(
        title=(
            f"{model_name} absolute-error distribution "
            f"within the ±{tolerance:.2f} subset"
        ),
        xlabel="Absolute error in logKp (cm/s)",
        ylabel="Number of measurements",
        xlim=(0, tolerance),
        ylim=(0, max(counts) * 1.15 if len(counts) else 1),
    )
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    add_selection_note(fig, tolerance)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(
    input_path: Path,
    output_dir: Path,
    experimental_column: str,
    prediction_column: str,
    tolerance: float,
    model_name: str = DEFAULT_MODEL_NAME,
    model_slug: str | None = None,
) -> tuple[Path, list[Path]]:
    """Create the filtered CSV and all figures, returning their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_slug = infer_model_slug(prediction_column, model_slug)
    filtered_prediction_column = f"Predicted_logKp_{resolved_slug}"
    absolute_error_column = f"Absolute_Error_{resolved_slug}"
    filtered = load_and_filter_predictions(
        input_path=input_path,
        experimental_column=experimental_column,
        prediction_column=prediction_column,
        tolerance=tolerance,
        model_slug=resolved_slug,
    )
    if filtered.empty:
        raise ValueError(f"No rows met the ±{tolerance:.4f} criterion.")

    tolerance_label = f"{tolerance:.4f}".rstrip("0").rstrip(".")
    csv_path = (
        output_dir
        / f"huskin_{resolved_slug}_predictions_within_{tolerance_label}.csv"
    )
    filtered.to_csv(csv_path, index=False, float_format="%.8f")

    configure_plot_style()
    plot_paths = [
        output_dir
        / f"experimental_vs_predicted_within_{tolerance_label}_{resolved_slug}.png",
        output_dir / f"residuals_within_{tolerance_label}_{resolved_slug}.png",
        output_dir
        / f"absolute_error_distribution_within_{tolerance_label}_{resolved_slug}.png",
    ]
    plot_experimental_vs_predicted(
        filtered,
        plot_paths[0],
        tolerance,
        model_name=model_name,
        prediction_column=filtered_prediction_column,
    )
    plot_residuals(
        filtered,
        plot_paths[1],
        tolerance,
        model_name=model_name,
        prediction_column=filtered_prediction_column,
    )
    plot_absolute_error_distribution(
        filtered,
        plot_paths[2],
        tolerance,
        model_name=model_name,
        absolute_error_column=absolute_error_column,
    )

    LOGGER.info(
        "Saved %d %s rows (%d unique canonical compounds) to %s.",
        len(filtered),
        model_name,
        filtered["Canonical_SMILES"].nunique(dropna=True),
        csv_path,
    )
    return csv_path, plot_paths


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Filter HuSkinDB predictions by tolerance and create plots."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experimental-column", default=DEFAULT_EXPERIMENTAL_COLUMN)
    parser.add_argument("--prediction-column", default=DEFAULT_PREDICTION_COLUMN)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-slug")
    return parser


def main() -> int:
    """Run the command-line interface."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    csv_path, plot_paths = run(
        input_path=args.input,
        output_dir=args.output,
        experimental_column=args.experimental_column,
        prediction_column=args.prediction_column,
        tolerance=args.tolerance,
        model_name=args.model_name,
        model_slug=args.model_slug,
    )
    LOGGER.info("Filtered CSV: %s", csv_path)
    for plot_path in plot_paths:
        LOGGER.info("Plot: %s", plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
