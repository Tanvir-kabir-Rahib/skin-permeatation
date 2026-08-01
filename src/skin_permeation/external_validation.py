from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_OUTPUT_DIR = Path("outputs") / "external_validation"
FINAL_STATEMENT = (
    "Interpret the metrics together with the saved validation protocol: only an independent "
    "or zero-overlap molecule-group holdout supports an unseen-compound predictivity claim."
)


def validate_columns(frame: pd.DataFrame, required_columns: list[str], source_name: str = "input CSV") -> None:
    """Raise a helpful error if any required columns are missing from a DataFrame."""
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        available = ", ".join(frame.columns.astype(str))
        missing_text = ", ".join(missing)
        raise ValueError(f"{source_name} is missing required column(s): {missing_text}. Available columns: {available}")


def load_prediction_data(csv_path: str | Path, actual_col: str, pred_col: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load external prediction data, validate required columns, and drop missing actual/predicted rows."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")

    frame = pd.read_csv(path)
    validate_columns(frame, [actual_col, pred_col], source_name=str(path))

    before_rows = len(frame)
    clean = frame.dropna(subset=[actual_col, pred_col]).copy()
    dropped = before_rows - len(clean)
    if dropped:
        warnings.warn(
            f"Removed {dropped} row(s) with missing values in '{actual_col}' or '{pred_col}'.",
            RuntimeWarning,
            stacklevel=2,
        )
    if clean.empty:
        raise ValueError("No valid rows remain after removing missing actual/predicted values.")

    try:
        y_true = clean[actual_col].astype(float).to_numpy()
        y_pred = clean[pred_col].astype(float).to_numpy()
    except ValueError as exc:
        raise ValueError(f"Columns '{actual_col}' and '{pred_col}' must contain numeric values.") from exc

    return clean, y_true, y_pred


def load_training_target(train_csv_path: str | Path | None, train_target_col: str | None) -> np.ndarray | None:
    """Load the training-set target values needed by Q2_F1 and Q2_F3, if a training CSV is supplied."""
    if train_csv_path is None:
        return None
    if not train_target_col:
        raise ValueError("train_target_col must be provided when train_csv_path is supplied.")

    path = Path(train_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    frame = pd.read_csv(path)
    validate_columns(frame, [train_target_col], source_name=str(path))
    target = frame[train_target_col].dropna()
    if target.empty:
        raise ValueError(f"No valid training target values remain in '{train_target_col}'.")
    try:
        return target.astype(float).to_numpy()
    except ValueError as exc:
        raise ValueError(f"Training target column '{train_target_col}' must contain numeric values.") from exc


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when the denominator is zero or invalid."""
    if not np.isfinite(denominator) or math.isclose(float(denominator), 0.0, abs_tol=1e-15):
        return float("nan")
    return float(numerator / denominator)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate R2_ext, RMSE_ext, and MAE_ext for external predictions."""
    return {
        "R2_ext": float(r2_score(y_true, y_pred)),
        "RMSE_ext": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_ext": float(mean_absolute_error(y_true, y_pred)),
    }


def calculate_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Lin's concordance correlation coefficient for actual and predicted values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    var_true = float(np.var(y_true, ddof=0))
    var_pred = float(np.var(y_pred, ddof=0))
    covariance = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return safe_divide(2.0 * covariance, denominator)


def calculate_q2_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray | None = None) -> dict[str, float]:
    """Calculate Q2_F1, Q2_F2, and Q2_F3 external validation metrics.

    Q2_F1 = 1 - PRESS / sum((y_ext - mean(y_train))^2)
    Q2_F2 = 1 - PRESS / sum((y_ext - mean(y_ext))^2)
    Q2_F3 = 1 - (PRESS / n_ext) / (sum((y_train - mean(y_train))^2) / n_train)

    PRESS is the prediction error sum of squares on the external set. Q2_F1 and Q2_F3
    need the training-set target distribution, so they are returned as NaN when y_train is missing.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    press = float(np.sum((y_true - y_pred) ** 2))
    external_denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))

    q2_f1 = float("nan")
    q2_f3 = float("nan")
    if y_train is None:
        warnings.warn(
            "Q2_F1 and Q2_F3 were set to NaN because training-set target values were not provided.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        y_train = np.asarray(y_train, dtype=float)
        train_mean = float(np.mean(y_train))
        q2_f1_denominator = float(np.sum((y_true - train_mean) ** 2))
        q2_f1 = 1.0 - safe_divide(press, q2_f1_denominator)

        q2_f3_denominator = safe_divide(float(np.sum((y_train - train_mean) ** 2)), float(len(y_train)))
        q2_f3 = 1.0 - safe_divide(press / float(len(y_true)), q2_f3_denominator)

    q2_f2 = 1.0 - safe_divide(press, external_denominator)

    return {
        "Q2_F1": float(q2_f1),
        "Q2_F2": float(q2_f2),
        "Q2_F3": float(q2_f3),
    }


def calculate_origin_forced_r2(response: np.ndarray, predictor: np.ndarray, slope: float) -> float:
    """Calculate R-squared for an origin-forced regression response ~= slope * predictor."""
    response = np.asarray(response, dtype=float)
    predictor = np.asarray(predictor, dtype=float)
    fitted = slope * predictor
    residual_sum_squares = float(np.sum((response - fitted) ** 2))
    total_sum_squares = float(np.sum((response - np.mean(response)) ** 2))
    return 1.0 - safe_divide(residual_sum_squares, total_sum_squares)


def calculate_golbraikh_tropsha(y_true: np.ndarray, y_pred: np.ndarray, r2_ext: float) -> dict[str, float | bool]:
    """Calculate Golbraikh-Tropsha origin-forced regression parameters and criteria."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    k = safe_divide(float(np.sum(y_true * y_pred)), float(np.sum(y_true**2)))
    k_prime = safe_divide(float(np.sum(y_true * y_pred)), float(np.sum(y_pred**2)))
    r0_sq = calculate_origin_forced_r2(response=y_pred, predictor=y_true, slope=k)
    r0_prime_sq = calculate_origin_forced_r2(response=y_true, predictor=y_pred, slope=k_prime)

    ratio_denominator = r2_ext if r2_ext > 0.0 else float("nan")
    r2_r0_ratio = safe_divide(abs(r2_ext - r0_sq), ratio_denominator)
    r2_r0_prime_ratio = safe_divide(abs(r2_ext - r0_prime_sq), ratio_denominator)
    gt_pass = (
        bool(r2_ext > 0.60)
        and bool(0.85 <= k <= 1.15)
        and bool(0.85 <= k_prime <= 1.15)
        and bool((r2_r0_ratio < 0.10) or (r2_r0_prime_ratio < 0.10))
    )

    return {
        "R0^2": float(r0_sq),
        "R0'^2": float(r0_prime_sq),
        "k": float(k),
        "k'": float(k_prime),
        "abs(R2_ext - R0^2) / R2_ext": float(r2_r0_ratio),
        "abs(R2_ext - R0'^2) / R2_ext": float(r2_r0_prime_ratio),
        "Golbraikh-Tropsha criteria": gt_pass,
    }


def calculate_rm2_metrics(y_true: np.ndarray, y_pred: np.ndarray, r0_sq: float, r0_prime_sq: float) -> dict[str, float]:
    """Calculate Roy r_m^2, reverse r_m'^2, average r_m^2, and delta r_m^2.

    r_m^2 = r^2 * (1 - sqrt(abs(r^2 - R0^2)))
    r_m'^2 = r^2 * (1 - sqrt(abs(r^2 - R0'^2)))

    Here r^2 is the squared Pearson correlation between observed and predicted values. The
    average and delta values are often reported to summarize symmetry between direct and
    reverse origin-forced fits.
    """
    if len(y_true) < 2 or math.isclose(float(np.std(y_true)), 0.0) or math.isclose(float(np.std(y_pred)), 0.0):
        correlation_sq = float("nan")
    else:
        correlation_matrix = np.corrcoef(y_true, y_pred)
        correlation_sq = float(correlation_matrix[0, 1] ** 2)

    rm2 = correlation_sq * (1.0 - math.sqrt(abs(correlation_sq - r0_sq)))
    rm2_prime = correlation_sq * (1.0 - math.sqrt(abs(correlation_sq - r0_prime_sq)))
    finite_rm2_values = [value for value in [rm2, rm2_prime] if np.isfinite(value)]
    rm2_average = float(np.mean(finite_rm2_values)) if finite_rm2_values else float("nan")
    delta_rm2 = abs(rm2 - rm2_prime)

    return {
        "r^2_pearson": float(correlation_sq),
        "r_m^2": float(rm2),
        "r_m'^2": float(rm2_prime),
        "Average r_m^2": float(rm2_average),
        "Delta r_m^2": float(delta_rm2),
    }


def calculate_external_validation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
) -> dict[str, float | bool]:
    """Calculate all supported external QSAR/QSPR validation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if len(y_true) < 2:
        raise ValueError("At least two valid external prediction rows are required for validation metrics.")

    regression = calculate_regression_metrics(y_true, y_pred)
    q2_metrics = calculate_q2_metrics(y_true, y_pred, y_train=y_train)
    ccc = calculate_ccc(y_true, y_pred)
    gt_metrics = calculate_golbraikh_tropsha(y_true, y_pred, r2_ext=regression["R2_ext"])
    rm2_metrics = calculate_rm2_metrics(
        y_true,
        y_pred,
        r0_sq=float(gt_metrics["R0^2"]),
        r0_prime_sq=float(gt_metrics["R0'^2"]),
    )

    return {
        **regression,
        **q2_metrics,
        "CCC_ext": float(ccc),
        **rm2_metrics,
        **gt_metrics,
    }


def pass_fail(value: float, criterion: str, passed: bool | None) -> str:
    """Convert a metric value and boolean criterion result into a report-friendly label."""
    if isinstance(value, float) and np.isnan(value):
        return "Not available"
    if passed is None:
        return "Report only"
    return "Pass" if passed else "Fail"


def build_summary_table(metrics: dict[str, float | bool]) -> pd.DataFrame:
    """Build the required validation summary table with acceptance criteria and pass/fail results."""
    ratio_pass = (
        float(metrics["R2_ext"]) > 0.0
        and (
            float(metrics["abs(R2_ext - R0^2) / R2_ext"]) < 0.10
            or float(metrics["abs(R2_ext - R0'^2) / R2_ext"]) < 0.10
        )
    )

    rows = [
        ("R2_ext", metrics["R2_ext"], "R2_ext > 0.60", float(metrics["R2_ext"]) > 0.60),
        ("RMSE_ext", metrics["RMSE_ext"], "Lower is better", None),
        ("MAE_ext", metrics["MAE_ext"], "Lower is better", None),
        ("Q2_F1", metrics["Q2_F1"], "Q2_F1 > 0.70", float(metrics["Q2_F1"]) > 0.70),
        ("Q2_F2", metrics["Q2_F2"], "Q2_F2 > 0.70", float(metrics["Q2_F2"]) > 0.70),
        ("Q2_F3", metrics["Q2_F3"], "Q2_F3 > 0.70", float(metrics["Q2_F3"]) > 0.70),
        ("CCC_ext", metrics["CCC_ext"], "CCC_ext > 0.85", float(metrics["CCC_ext"]) > 0.85),
        ("r_m^2", metrics["r_m^2"], "r_m^2 > 0.65", float(metrics["r_m^2"]) > 0.65),
        ("r_m'^2", metrics["r_m'^2"], "Report alongside r_m^2", None),
        ("Average r_m^2", metrics["Average r_m^2"], "Average r_m^2 > 0.65", float(metrics["Average r_m^2"]) > 0.65),
        ("Delta r_m^2", metrics["Delta r_m^2"], "Delta r_m^2 < 0.20", float(metrics["Delta r_m^2"]) < 0.20),
        ("R0^2", metrics["R0^2"], "Report for Golbraikh-Tropsha", None),
        ("R0'^2", metrics["R0'^2"], "Report for Golbraikh-Tropsha", None),
        ("k", metrics["k"], "0.85 <= k <= 1.15", 0.85 <= float(metrics["k"]) <= 1.15),
        ("k'", metrics["k'"], "0.85 <= k' <= 1.15", 0.85 <= float(metrics["k'"]) <= 1.15),
        (
            "abs(R2_ext - R0^2) / R2_ext",
            metrics["abs(R2_ext - R0^2) / R2_ext"],
            "< 0.10 OR reverse ratio < 0.10",
            ratio_pass,
        ),
        (
            "abs(R2_ext - R0'^2) / R2_ext",
            metrics["abs(R2_ext - R0'^2) / R2_ext"],
            "< 0.10 OR direct ratio < 0.10",
            ratio_pass,
        ),
        (
            "Golbraikh-Tropsha criteria",
            float(metrics["Golbraikh-Tropsha criteria"]),
            "R2_ext > 0.60, slopes in range, and one ratio < 0.10",
            bool(metrics["Golbraikh-Tropsha criteria"]),
        ),
    ]

    formatted_rows = []
    for metric, value, criterion, passed in rows:
        numeric_value = float(value)
        formatted_rows.append(
            {
                "Metric": metric,
                "Value": numeric_value,
                "Acceptance Criterion": criterion,
                "Result": pass_fail(numeric_value, criterion, passed),
            }
        )
    return pd.DataFrame(formatted_rows)


def build_journal_table(summary_table: pd.DataFrame) -> pd.DataFrame:
    """Build a journal-ready validation table with threshold and interpretation columns."""
    journal = summary_table.rename(
        columns={
            "Acceptance Criterion": "Recommended threshold",
            "Result": "Interpretation",
        }
    ).copy()
    report_only_text = {
        "RMSE_ext": "Prediction-error metric; lower values indicate better predictions",
        "MAE_ext": "Prediction-error metric; lower values indicate better predictions",
        "r_m'^2": "Reverse-axis modified correlation reported alongside r_m^2; higher values are better",
        "R0^2": "Origin-forced direct-fit statistic used by the Golbraikh-Tropsha assessment",
        "R0'^2": "Origin-forced reverse-fit statistic used by the Golbraikh-Tropsha assessment",
    }

    def interpretation(row: pd.Series) -> str:
        result = str(row["Interpretation"])
        if result == "Pass":
            return "Satisfies the recommended criterion"
        if result == "Fail":
            return "Does not satisfy the recommended criterion"
        if result == "Not available":
            return "Not available because the required training-set information was not provided"
        return report_only_text.get(str(row["Metric"]), "Supporting validation statistic")

    journal["Interpretation"] = journal.apply(interpretation, axis=1)
    return journal[["Metric", "Value", "Recommended threshold", "Interpretation"]]


def print_validation_report(summary_table: pd.DataFrame, journal_table: pd.DataFrame, output_dir: Path) -> None:
    """Print a terminal validation report and the journal-ready table."""
    print("\nExternal QSAR/QSPR Validation Report")
    print("=" * 39)
    print(summary_table.to_string(index=False))
    print("\nJournal-ready validation table")
    print("=" * 30)
    print(journal_table.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir.resolve()}")
    print(f"\n{FINAL_STATEMENT}")


def plot_experimental_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float | bool],
    output_path: str | Path,
    x_label: str = "Experimental logKp",
    y_label: str = "Predicted logKp",
    title: str = "Validation predictions: experimental vs predicted logKp",
) -> Path:
    """Create and save a publication-quality predicted-vs-experimental scatter plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(
        y_true,
        y_pred,
        s=48,
        color="#2563a6",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.88,
        label="Validation observations",
    )

    xy_min = float(np.nanmin([np.min(y_true), np.min(y_pred)]))
    xy_max = float(np.nanmax([np.max(y_true), np.max(y_pred)]))
    margin = 0.06 * (xy_max - xy_min if xy_max > xy_min else 1.0)
    plot_min = xy_min - margin
    plot_max = xy_max + margin
    line_x = np.linspace(plot_min, plot_max, 200)

    ax.plot(line_x, line_x, color="#222222", linestyle="--", linewidth=1.3, label="Ideal agreement")
    slope, intercept = np.polyfit(y_true, y_pred, deg=1)
    ax.plot(line_x, slope * line_x + intercept, color="#d58a19", linewidth=1.7, label="Fitted regression")

    text = (
        f"n = {len(y_true)}\n"
        f"R² = {float(metrics['R2_ext']):.3f}\n"
        f"RMSE = {float(metrics['RMSE_ext']):.3f}\n"
        f"MAE = {float(metrics['MAE_ext']):.3f}\n"
        f"CCC = {float(metrics['CCC_ext']):.3f}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#444444", "alpha": 0.95},
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Plot residuals against fitted values and their distribution."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    scatter_axis, histogram_axis = axes
    scatter_axis.axhspan(-0.5, 0.5, color="#2563a6", alpha=0.08, label="±0.5 logKp")
    scatter_axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1.2)
    scatter_axis.scatter(
        y_pred,
        residuals,
        s=42,
        color="#2563a6",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.86,
    )
    scatter_axis.set_xlabel("Predicted logKp")
    scatter_axis.set_ylabel("Residual (predicted − experimental)")
    scatter_axis.set_title("Residuals versus predictions")
    scatter_axis.legend(frameon=False, loc="upper right")
    scatter_axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.70)
    scatter_axis.set_axisbelow(True)

    histogram_axis.hist(residuals, bins="auto", color="#2563a6", edgecolor="white", linewidth=0.8, alpha=0.88)
    histogram_axis.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2)
    histogram_axis.axvline(float(np.mean(residuals)), color="#d58a19", linewidth=1.6, label="Mean residual")
    histogram_axis.set_xlabel("Residual (predicted − experimental)")
    histogram_axis.set_ylabel("Observation count")
    histogram_axis.set_title("Residual distribution")
    histogram_axis.legend(frameon=False, loc="upper left")
    histogram_axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.70)
    histogram_axis.set_axisbelow(True)

    fig.suptitle(f"Validation residual diagnostics (n={len(y_true)})", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def calculate_criteria_margins(metrics: dict[str, float | bool]) -> list[tuple[str, float, str]]:
    """Return normalized margins from the project acceptance cutoffs for plotting."""

    def lower_margin(value: float, cutoff: float) -> float:
        return value / cutoff - 1.0

    def upper_margin(value: float, cutoff: float) -> float:
        return 1.0 - value / cutoff

    def bounded_margin(value: float, lower: float, upper: float) -> float:
        half_width = (upper - lower) / 2.0
        return min((value - lower) / half_width, (upper - value) / half_width)

    ratio = min(
        float(metrics["abs(R2_ext - R0^2) / R2_ext"]),
        float(metrics["abs(R2_ext - R0'^2) / R2_ext"]),
    )
    k_prime = float(metrics["k'"])
    return [
        ("R²ext > 0.60", lower_margin(float(metrics["R2_ext"]), 0.60), f"{float(metrics['R2_ext']):.3f}"),
        ("Q²F1 > 0.70", lower_margin(float(metrics["Q2_F1"]), 0.70), f"{float(metrics['Q2_F1']):.3f}"),
        ("Q²F2 > 0.70", lower_margin(float(metrics["Q2_F2"]), 0.70), f"{float(metrics['Q2_F2']):.3f}"),
        ("Q²F3 > 0.70", lower_margin(float(metrics["Q2_F3"]), 0.70), f"{float(metrics['Q2_F3']):.3f}"),
        ("CCC > 0.85", lower_margin(float(metrics["CCC_ext"]), 0.85), f"{float(metrics['CCC_ext']):.3f}"),
        ("rₘ² > 0.65", lower_margin(float(metrics["r_m^2"]), 0.65), f"{float(metrics['r_m^2']):.3f}"),
        ("Average rₘ² > 0.65", lower_margin(float(metrics["Average r_m^2"]), 0.65), f"{float(metrics['Average r_m^2']):.3f}"),
        ("Δrₘ² < 0.20", upper_margin(float(metrics["Delta r_m^2"]), 0.20), f"{float(metrics['Delta r_m^2']):.3f}"),
        ("0.85 ≤ k ≤ 1.15", bounded_margin(float(metrics["k"]), 0.85, 1.15), f"{float(metrics['k']):.3f}"),
        ("0.85 ≤ k′ ≤ 1.15", bounded_margin(k_prime, 0.85, 1.15), f"{k_prime:.3f}"),
        ("Best GT ratio < 0.10", upper_margin(ratio, 0.10), f"{ratio:.3f}"),
    ]


def plot_validation_criteria(metrics: dict[str, float | bool], output_path: str | Path) -> Path:
    """Plot each thresholded metric as its relative margin from the acceptance cutoff."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = calculate_criteria_margins(metrics)
    labels = [row[0] for row in rows]
    margins = np.asarray([row[1] for row in rows], dtype=float)
    values = [row[2] for row in rows]
    colors = ["#2563a6" if margin > 0.0 else "#d58a19" for margin in margins]

    fig, ax = plt.subplots(figsize=(8.7, 6.6))
    positions = np.arange(len(rows))
    ax.barh(positions, margins * 100.0, color=colors, edgecolor="#243240", linewidth=0.5)
    ax.axvline(0.0, color="#222222", linewidth=1.2)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relative margin from acceptance cutoff (%)")
    ax.set_title(
        "Positive margins pass; labels show the metric value",
        loc="left",
        color="#4a5560",
        fontsize=10,
        pad=10,
    )
    fig.suptitle("Validation acceptance criteria", y=0.985, fontsize=13)
    margin_percent = margins * 100.0
    x_min = min(-5.0, float(np.nanmin(margin_percent)) - 5.0)
    x_max = max(10.0, float(np.nanmax(margin_percent)) + 12.0)
    padding = 0.02 * (x_max - x_min)
    for position, (margin, value) in enumerate(zip(margin_percent, values)):
        if margin >= 0.0:
            ax.text(margin + padding, position, value, va="center", ha="left", fontsize=9)
        else:
            ax.text(margin - padding, position, value, va="center", ha="right", fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.grid(True, axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.70)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_validation_outputs(summary_table: pd.DataFrame, journal_table: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Save validation tables as CSV and Excel files in the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "external_validation_metrics.csv"
    summary_xlsx = output_dir / "external_validation_metrics.xlsx"
    journal_csv = output_dir / "journal_ready_external_validation_table.csv"
    journal_xlsx = output_dir / "journal_ready_external_validation_table.xlsx"

    summary_table.to_csv(summary_csv, index=False)
    summary_table.to_excel(summary_xlsx, index=False)
    journal_table.to_csv(journal_csv, index=False)
    journal_table.to_excel(journal_xlsx, index=False)

    return {
        "summary_csv": summary_csv,
        "summary_xlsx": summary_xlsx,
        "journal_csv": journal_csv,
        "journal_xlsx": journal_xlsx,
    }


def run_external_validation(
    csv_path: str | Path,
    actual_col: str = "Experimental_logKp",
    pred_col: str = "Predicted_logKp",
    train_csv_path: str | Path | None = None,
    train_target_col: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Run the full external validation workflow and return the summary metrics table.

    The workflow loads predictions, removes rows with missing actual/predicted values, calculates
    external QSAR/QSPR metrics, prints a validation report, saves summary and journal-ready tables,
    and writes a publication-quality experimental-vs-predicted scatter plot.
    """
    try:
        _, y_true, y_pred = load_prediction_data(csv_path, actual_col=actual_col, pred_col=pred_col)
        y_train = load_training_target(train_csv_path, train_target_col)
        metrics = calculate_external_validation_metrics(y_true, y_pred, y_train=y_train)
        summary_table = build_summary_table(metrics)
        journal_table = build_journal_table(summary_table)

        output_dir = Path(output_dir)
        save_validation_outputs(summary_table, journal_table, output_dir)
        plot_experimental_vs_predicted(
            y_true,
            y_pred,
            metrics,
            output_path=output_dir / "experimental_vs_predicted_logKp.png",
        )
        plot_residual_diagnostics(
            y_true,
            y_pred,
            output_path=output_dir / "external_validation_residual_diagnostics.png",
        )
        plot_validation_criteria(
            metrics,
            output_path=output_dir / "external_validation_criteria_summary.png",
        )
        print_validation_report(summary_table, journal_table, output_dir)
        return summary_table
    except Exception as exc:
        print(f"External validation failed: {exc}")
        raise


if __name__ == "__main__":
    run_external_validation(
        csv_path="external_test_predictions.csv",
        actual_col="Experimental_logKp",
        pred_col="Predicted_logKp",
        train_csv_path="train_data.csv",
        train_target_col="Experimental_logKp",
    )
