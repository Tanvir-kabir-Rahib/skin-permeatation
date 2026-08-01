from __future__ import annotations

import json
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from .external_validation import (
    build_summary_table,
    calculate_external_validation_metrics,
    plot_experimental_vs_predicted,
)
from .validation_model import (
    load_modeling_dataset,
    prepare_modeling_data,
    resolve_validation_protocol,
    split_external_holdout,
)


MODEL_FILE_NAMES = {
    "Gradient Boosting": "Gradient_Boosting.joblib",
    "LGBM": "LGBM.joblib",
    "ExtraTrees": "ExtraTrees.joblib",
    "XGBoost": "XGBoost.joblib",
    "HistGradientBoosting": "HistGradientBoosting.joblib",
    "Mean Ensemble": "MeanEnsemble.joblib",
    "Stacking Regressor": "StackingRegressor.joblib",
    "SVR (RBF)": "SVR_RBF.joblib",
    "ElasticNet": "ElasticNet.joblib",
    "CatBoost": "CatBoost.joblib",
    "RF": "RF.joblib",
}

MODEL_DISPLAY_NAMES = {Path(file_name).stem: name for name, file_name in MODEL_FILE_NAMES.items()}
THRESHOLDED_METRICS = [
    "R2_ext",
    "Q2_F1",
    "Q2_F2",
    "Q2_F3",
    "CCC_ext",
    "r_m^2",
    "Average r_m^2",
    "Delta r_m^2",
    "k",
    "k'",
    "abs(R2_ext - R0^2) / R2_ext",
    "abs(R2_ext - R0'^2) / R2_ext",
    "Golbraikh-Tropsha criteria",
]


@dataclass
class BenchmarkExternalValidationResult:
    """Tables and paths produced by the all-benchmark validation workflow."""

    summary: pd.DataFrame
    metrics_long: pd.DataFrame
    predictions_long: pd.DataFrame
    protocol: dict[str, Any]
    output_dir: Path


def _slugify_model_name(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
        .replace(" ", "_")
    )


def _artifact_for_model(model_name: str, artifact_value: str, models_dir: Path) -> Path:
    preferred = models_dir / MODEL_FILE_NAMES.get(model_name, Path(artifact_value).name)
    if preferred.exists():
        return preferred
    candidate = Path(artifact_value)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Benchmark artifact for '{model_name}' was not found: {preferred}")


def load_benchmark_inventory(
    benchmark_metrics_path: str | Path,
    models_dir: str | Path,
) -> pd.DataFrame:
    """Resolve the unique benchmark artifacts in report order."""
    metrics_path = Path(benchmark_metrics_path)
    models_path = Path(models_dir)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Benchmark metrics table not found: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    required = {"model", "artifact_path"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Benchmark metrics table is missing columns: {', '.join(sorted(missing))}")

    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for row in metrics.itertuples(index=False):
        model_name = str(row.model)
        artifact = _artifact_for_model(model_name, str(row.artifact_path), models_path).resolve()
        if artifact.name == "best_benchmark_model.joblib" or artifact in seen:
            continue
        rows.append({"Model": model_name, "Source Artifact": str(artifact)})
        seen.add(artifact)

    for artifact in sorted(models_path.glob("*.joblib")):
        resolved = artifact.resolve()
        if artifact.name == "best_benchmark_model.joblib" or resolved in seen:
            continue
        model_name = MODEL_DISPLAY_NAMES.get(artifact.stem, artifact.stem.replace("_", " "))
        rows.append({"Model": model_name, "Source Artifact": str(resolved)})
        seen.add(resolved)
    if not rows:
        raise ValueError("No benchmark model artifacts were discovered.")
    return pd.DataFrame(rows)


def _assert_feature_compatibility(estimator: Any, feature_columns: list[str], model_name: str) -> None:
    expected = list(getattr(estimator, "feature_names_in_", []))
    if expected and expected != feature_columns:
        raise ValueError(
            f"{model_name} expects {len(expected)} feature columns, but the validation dataset "
            f"provides a different ordered set of {len(feature_columns)} columns."
        )


def _validation_overlap_summary(
    identifiers: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> dict[str, int | None]:
    smiles_column = next((column for column in ("SMILES", "smiles") if column in identifiers.columns), None)
    if smiles_column is None:
        return {
            "unique_smiles": None,
            "training_unique_smiles": None,
            "validation_unique_smiles": None,
            "overlapping_smiles": None,
            "overlapping_validation_rows": None,
        }
    smiles = identifiers[smiles_column].fillna("__missing_smiles__").astype(str)
    train_smiles = set(smiles.iloc[train_indices])
    validation_smiles = set(smiles.iloc[validation_indices])
    overlap = train_smiles & validation_smiles
    return {
        "unique_smiles": int(smiles.nunique()),
        "training_unique_smiles": int(len(train_smiles)),
        "validation_unique_smiles": int(len(validation_smiles)),
        "overlapping_smiles": int(len(overlap)),
        "overlapping_validation_rows": int(smiles.iloc[validation_indices].isin(overlap).sum()),
    }


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_performance_comparison(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Compare agreement and error metrics across all validation models."""
    output_path = Path(output_path)
    ordered = summary.sort_values(["RMSE_ext", "R2_ext"], ascending=[True, False]).reset_index(drop=True)
    positions = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), gridspec_kw={"width_ratios": [1.05, 1.0]})

    agreement = axes[0]
    height = 0.36
    agreement.barh(positions - height / 2, ordered["R2_ext"], height=height, color="#2563a6", label="R²_ext")
    agreement.barh(
        positions + height / 2,
        ordered["CCC_ext"],
        height=height,
        color="#d58a19",
        label="CCC_ext",
    )
    agreement.axvline(0.60, color="#2563a6", linestyle="--", linewidth=1.1, alpha=0.75)
    agreement.axvline(0.85, color="#d58a19", linestyle=":", linewidth=1.3, alpha=0.85)
    agreement.set_yticks(positions, ordered["Model"])
    agreement.invert_yaxis()
    agreement.set_xlim(0.0, 1.02)
    agreement.set_xlabel("Metric value")
    agreement.set_title("Agreement metrics")
    agreement.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.75)

    error_axis = axes[1]
    error_axis.barh(positions - height / 2, ordered["RMSE_ext"], height=height, color="#2563a6", label="RMSE_ext")
    error_axis.barh(positions + height / 2, ordered["MAE_ext"], height=height, color="#9aa3ad", label="MAE_ext")
    error_axis.set_yticks(positions, [])
    error_axis.invert_yaxis()
    error_axis.set_xlim(left=0.0)
    error_axis.set_xlabel("Prediction error (logKp; lower is better)")
    error_axis.set_title("Prediction-error metrics")
    error_axis.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.75)
    handles = [
        agreement.patches[0],
        agreement.patches[len(ordered)],
        error_axis.patches[0],
        error_axis.patches[len(ordered)],
    ]
    labels = ["R²_ext", "CCC_ext", "RMSE_ext", "MAE_ext"]

    fig.suptitle("External-validation performance across benchmark models", fontsize=14, y=0.995)
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.text(
        0.5,
        0.012,
        "Common paper-reproduction holdout: n=63; dashed/dotted agreement references show configured cutoffs.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.tight_layout(rect=[0.0, 0.045, 1.0, 0.90])
    return _save_figure(fig, output_path)


def plot_criteria_pass_counts(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot how many of the 13 thresholded criteria each model satisfies."""
    output_path = Path(output_path)
    ordered = summary.sort_values(
        ["Criteria Passed", "RMSE_ext"], ascending=[False, True]
    ).reset_index(drop=True)
    colors = ["#2563a6" if passed == len(THRESHOLDED_METRICS) else "#d58a19" for passed in ordered["Criteria Passed"]]
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    bars = ax.barh(ordered["Model"], ordered["Criteria Passed"], color=colors, edgecolor="#374151", linewidth=0.45)
    ax.invert_yaxis()
    ax.set_xlim(0, len(THRESHOLDED_METRICS) + 0.7)
    ax.set_xticks(range(0, len(THRESHOLDED_METRICS) + 1))
    ax.set_xlabel(f"Criteria passed (out of {len(THRESHOLDED_METRICS)})")
    ax.set_title("External-validation acceptance criteria by benchmark model")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.75)
    for bar, value in zip(bars, ordered["Criteria Passed"], strict=True):
        ax.text(value + 0.16, bar.get_y() + bar.get_height() / 2, f"{int(value)}/{len(THRESHOLDED_METRICS)}", va="center", fontsize=9)
    fig.text(
        0.5,
        -0.015,
        "Blue = all configured thresholds satisfied; orange = one or more thresholds not satisfied.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_criteria_matrix(metrics_long: pd.DataFrame, output_path: str | Path) -> Path:
    """Render a model-by-criterion pass/fail matrix with direct state labels."""
    output_path = Path(output_path)
    thresholded = metrics_long[metrics_long["Metric"].isin(THRESHOLDED_METRICS)].copy()
    model_order = (
        thresholded.assign(Passed=thresholded["Result"].eq("Pass").astype(int))
        .groupby("Model", sort=False)["Passed"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    state = thresholded.pivot(index="Model", columns="Metric", values="Result").reindex(
        index=model_order, columns=THRESHOLDED_METRICS
    )
    matrix = state.apply(lambda column: column.map({"Fail": 0.0, "Pass": 1.0})).to_numpy(dtype=float)
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(15.0, 6.0))
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(["#f3c47b", "#7ca6d8"]), vmin=0, vmax=1)
    short_labels = [
        "R²",
        "Q²F1",
        "Q²F2",
        "Q²F3",
        "CCC",
        "rₘ²",
        "Avg rₘ²",
        "Δrₘ²",
        "k",
        "k′",
        "GT ratio",
        "GT reverse ratio",
        "GT overall",
    ]
    ax.set_xticks(np.arange(len(short_labels)), short_labels, rotation=42, ha="right")
    ax.set_yticks(np.arange(len(model_order)), model_order)
    ax.set_title("Pass/fail matrix for external-validation criteria")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            label = "Pass" if matrix[row, column] == 1 else "Fail"
            ax.text(column, row, label, ha="center", va="center", fontsize=7.5, color="#111827")
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(
        0.5,
        -0.03,
        "Each cell uses the acceptance rule recorded in the metrics table; the two GT-ratio cells share the configured OR rule.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.tight_layout()
    return _save_figure(fig, output_path)


def validate_all_benchmark_models(
    dataset_csv_path: str | Path = "data/final/clean_trial4.csv",
    target_col: str | None = "logkpl",
    protocol: str = "paper-reproduction",
    split_strategy: str | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    benchmark_metrics_path: str | Path = "reports/tables/benchmark_metrics.csv",
    models_dir: str | Path = "models/reproduction/benchmark",
    output_dir: str | Path = "outputs/external_validation/benchmark_models",
    model_names: list[str] | None = None,
    save_fitted_models: bool = True,
    make_plots: bool = True,
) -> BenchmarkExternalValidationResult:
    """Retrain every benchmark estimator on one frozen holdout and validate each model."""
    split_strategy, test_size, random_state = resolve_validation_protocol(
        protocol, split_strategy, test_size, random_state
    )
    frame, resolved_target = load_modeling_dataset(dataset_csv_path, target_col=target_col)
    features, y, identifiers = prepare_modeling_data(frame, resolved_target)
    train_indices, validation_indices = split_external_holdout(
        features,
        y,
        identifiers,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
    )
    inventory = load_benchmark_inventory(benchmark_metrics_path, models_dir)
    if model_names is not None:
        inventory = inventory[inventory["Model"].isin(model_names)].reset_index(drop=True)
        missing_names = sorted(set(model_names).difference(inventory["Model"]))
        if missing_names:
            raise ValueError(f"Requested benchmark model(s) were not found: {', '.join(missing_names)}")
    if inventory.empty:
        raise ValueError("No benchmark models remain after filtering.")

    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    fitted_models_dir = output_path / "retrained_models"
    output_path.mkdir(parents=True, exist_ok=True)
    if make_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    if save_fitted_models:
        fitted_models_dir.mkdir(parents=True, exist_ok=True)

    x_train = features.iloc[train_indices]
    x_validation = features.iloc[validation_indices]
    y_train = y[train_indices]
    y_validation = y[validation_indices]
    feature_columns = list(features.columns)

    metric_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    for inventory_row in inventory.to_dict(orient="records"):
        model_name = str(inventory_row["Model"])
        source_artifact = Path(str(inventory_row["Source Artifact"]))
        template = joblib.load(source_artifact)
        _assert_feature_compatibility(template, feature_columns, model_name)
        estimator = clone(template)
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
            estimator.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        predictions = np.asarray(estimator.predict(x_validation), dtype=float)
        if predictions.shape != y_validation.shape or not np.isfinite(predictions).all():
            raise ValueError(f"{model_name} produced invalid validation predictions.")

        metrics = calculate_external_validation_metrics(y_validation, predictions, y_train=y_train)
        metric_table = build_summary_table(metrics)
        metric_table.insert(0, "Model", model_name)
        metric_tables.append(metric_table)

        thresholded = metric_table[metric_table["Result"].isin(["Pass", "Fail"])]
        passed = int(thresholded["Result"].eq("Pass").sum())
        failed = int(thresholded["Result"].eq("Fail").sum())
        retrained_artifact = fitted_models_dir / source_artifact.name
        if save_fitted_models:
            joblib.dump(estimator, retrained_artifact)

        summary_rows.append(
            {
                "Model": model_name,
                "R2_ext": float(metrics["R2_ext"]),
                "RMSE_ext": float(metrics["RMSE_ext"]),
                "MAE_ext": float(metrics["MAE_ext"]),
                "Q2_F1": float(metrics["Q2_F1"]),
                "Q2_F2": float(metrics["Q2_F2"]),
                "Q2_F3": float(metrics["Q2_F3"]),
                "CCC_ext": float(metrics["CCC_ext"]),
                "r_m^2": float(metrics["r_m^2"]),
                "Average r_m^2": float(metrics["Average r_m^2"]),
                "Delta r_m^2": float(metrics["Delta r_m^2"]),
                "k": float(metrics["k"]),
                "k'": float(metrics["k'"]),
                "Criteria Passed": passed,
                "Criteria Failed": failed,
                "All Criteria Pass": failed == 0,
                "Fit Seconds": float(fit_seconds),
                "Source Artifact": str(source_artifact.resolve()),
                "Retrained Artifact": str(retrained_artifact.resolve()) if save_fitted_models else "",
            }
        )
        training_rows.append(
            {
                "Model": model_name,
                "Source Artifact": str(source_artifact.resolve()),
                "Estimator Class": type(template).__name__,
                "Fit Seconds": float(fit_seconds),
                "Training Rows": int(len(train_indices)),
                "Validation Rows": int(len(validation_indices)),
            }
        )

        model_predictions = identifiers.iloc[validation_indices].reset_index(drop=True).copy()
        model_predictions.insert(0, "Validation Row", validation_indices.astype(int))
        model_predictions.insert(0, "Model", model_name)
        model_predictions["Experimental_logKp"] = y_validation
        model_predictions["Predicted_logKp"] = predictions
        model_predictions["Residual"] = predictions - y_validation
        model_predictions["Absolute Error"] = np.abs(predictions - y_validation)
        prediction_tables.append(model_predictions)

        if make_plots:
            slug = _slugify_model_name(model_name)
            plot_experimental_vs_predicted(
                y_validation,
                predictions,
                metrics,
                plots_dir / f"{slug}_experimental_vs_predicted.png",
                title=f"{model_name}: experimental vs predicted logKp",
            )

    metrics_long = pd.concat(metric_tables, ignore_index=True)
    predictions_long = pd.concat(prediction_tables, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["Criteria Passed", "RMSE_ext", "R2_ext"], ascending=[False, True, False]
    ).reset_index(drop=True)
    training_log = pd.DataFrame(training_rows)

    prediction_ids = [column for column in ("Validation Row", "Compound", "SMILES", "Experimental_logKp") if column in predictions_long.columns]
    predictions_wide = predictions_long.pivot_table(
        index=prediction_ids,
        columns="Model",
        values="Predicted_logKp",
        aggfunc="first",
    ).reset_index()
    predictions_wide.columns.name = None

    overlap_summary = _validation_overlap_summary(identifiers, train_indices, validation_indices)
    protocol_record: dict[str, Any] = {
        "protocol": protocol,
        "protocol_interpretation": (
            "Strict molecule-group holdout with no SMILES overlap"
            if split_strategy == "grouped"
            else "Paper-reproduction random row holdout; not strict unseen-molecule external validation"
        ),
        "dataset": {
            "path": str(Path(dataset_csv_path).resolve()),
            "rows": int(len(frame)),
            "feature_columns": int(len(feature_columns)),
            "target_column": resolved_target,
            "target_missing_rows": int(frame[resolved_target].isna().sum()),
            "feature_missing_cells": int(features.isna().sum().sum()),
            "exact_duplicate_rows": int(frame.duplicated().sum()),
            "unique_smiles": overlap_summary["unique_smiles"],
        },
        "split": {
            "strategy": split_strategy,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "training_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "training_unique_smiles": overlap_summary["training_unique_smiles"],
            "validation_unique_smiles": overlap_summary["validation_unique_smiles"],
            "overlapping_smiles": overlap_summary["overlapping_smiles"],
            "overlapping_validation_rows": overlap_summary["overlapping_validation_rows"],
        },
        "models": {
            "count": int(len(summary)),
            "names": inventory["Model"].tolist(),
            "templates_retrained_on_common_training_rows": True,
            "external_labels_used_for_training_or_tuning": False,
            "best_model_selected_after_validation": False,
        },
        "acceptance_summary": {
            "criteria_per_model": int(len(THRESHOLDED_METRICS)),
            "models_passing_all_criteria": int(summary["All Criteria Pass"].sum()),
            "models_failing_one_or_more_criteria": int((~summary["All Criteria Pass"]).sum()),
        },
        "software": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    summary.to_csv(output_path / "benchmark_external_validation_summary.csv", index=False)
    metrics_long.to_csv(output_path / "benchmark_external_validation_metrics_long.csv", index=False)
    predictions_long.to_csv(output_path / "benchmark_external_validation_predictions_long.csv", index=False)
    predictions_wide.to_csv(output_path / "benchmark_external_validation_predictions_wide.csv", index=False)
    training_log.to_csv(output_path / "benchmark_external_validation_training_log.csv", index=False)
    (output_path / "benchmark_external_validation_protocol.json").write_text(
        json.dumps(protocol_record, indent=2), encoding="utf-8"
    )

    if make_plots:
        plot_model_performance_comparison(summary, plots_dir / "benchmark_model_performance_comparison.png")
        plot_criteria_pass_counts(summary, plots_dir / "benchmark_model_criteria_pass_counts.png")
        plot_criteria_matrix(metrics_long, plots_dir / "benchmark_model_criteria_matrix.png")

    return BenchmarkExternalValidationResult(
        summary=summary,
        metrics_long=metrics_long,
        predictions_long=predictions_long,
        protocol=protocol_record,
        output_dir=output_path,
    )
