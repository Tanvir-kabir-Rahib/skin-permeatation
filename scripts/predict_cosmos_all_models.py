"""Predict COSMOS logKp with every compatible benchmark model and plot comparisons.

Saved models predict log10(cm/h); COSMOS reports log10(cm/s), so predictions
are converted by subtracting log10(3600). The requested "nearest experimental"
value is selected independently for every compound/model prediction from the
pipe-delimited experimental measurements. Mean-based errors are also retained
to make the optimistic nearest-value comparison transparent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/skin-cosmos-mpl")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_permeation.huskin import build_descriptor_audit, prepare_descriptor_frame
from skin_permeation.paths import ProjectPaths
from skin_permeation.prediction import load_available_models, predict_with_model

DEFAULT_INPUT = Path("data/cosmosDB/COSMOS_53_compounds_with_experimental_logKp.csv")
DEFAULT_OUTPUT = Path("results/cosmosDB")
SMILES_COLUMN = "SMILES"
TARGET_MEAN = "Experimental_logKp_mean_log10_cm_per_s"
TARGET_VALUES = "Experimental_logKp_values_log10_cm_per_s"
PREDICTION_OFFSET = -math.log10(3600.0)

# Exact/near-exact semantic aliases in the supplied COSMOS descriptor panel.
# Copying these into the fitted feature names lets the shared descriptor
# pipeline reuse source values before RDKit/CDK fills the remaining features.
COSMOS_DESCRIPTOR_ALIASES = {
    "XlogP": "XLogP",
    "TPSA": "TopoPSA",
    "Weight": "MW",
    "HAcc": "nHBAcc",
    "HDon": "nHBDon",
    "BondsRot": "nRotB",
    "Ro5Viol": "LipinskiFailures",
}


@dataclass(frozen=True)
class FeatureRequest:
    required_features: tuple[str, ...]
    display_name: str = "All compatible saved models"


def parse_experimental_values(value: object) -> list[float]:
    if pd.isna(value):
        return []
    values = []
    for token in re.split(r"\s*[|;,]\s*", str(value).strip()):
        if not token:
            continue
        try:
            number = float(token)
        except ValueError:
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def nearest_experimental(value: object, prediction: float) -> float:
    values = parse_experimental_values(value)
    if not values or not math.isfinite(prediction):
        return math.nan
    return min(values, key=lambda item: (abs(item - prediction), item))


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    pair = pd.DataFrame({"actual": actual, "predicted": predicted}).replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty:
        return {"n": 0, "R2": math.nan, "RMSE": math.nan, "MAE": math.nan, "Bias": math.nan}
    error = pair["predicted"] - pair["actual"]
    ss_tot = float(((pair["actual"] - pair["actual"].mean()) ** 2).sum())
    ss_res = float((error**2).sum())
    return {
        "n": len(pair),
        "R2": 1.0 - ss_res / ss_tot if len(pair) > 1 and ss_tot else math.nan,
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def make_plots(predictions: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white", "axes.facecolor": "white"})
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model, group in predictions.groupby("Model", sort=False):
        valid = group.dropna(subset=["Nearest_Experimental_logKp", "Predicted_logKp_cm_per_s"])
        row = metrics.loc[metrics["Model"] == model].iloc[0]
        finite = pd.concat([valid["Nearest_Experimental_logKp"], valid["Predicted_logKp_cm_per_s"]])
        lo, hi = float(finite.min()), float(finite.max())
        pad = max(0.25, (hi - lo) * 0.04)
        limits = (lo - pad, hi + pad)
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        ax.scatter(valid["Nearest_Experimental_logKp"], valid["Predicted_logKp_cm_per_s"], s=34, alpha=0.72,
                   color="#2563eb", edgecolors="white", linewidths=0.4)
        ax.plot(limits, limits, color="#334155", linestyle="--", linewidth=1.2, label="Ideal agreement")
        ax.set(xlim=limits, ylim=limits, xlabel="Nearest experimental logKp (log10 cm/s)",
               ylabel="Predicted logKp (log10 cm/s)", title=f"COSMOS predicted vs nearest experimental logKp\n{model}")
        ax.text(0.03, 0.97, f"n={int(row['n'])}   RMSE={row['RMSE']:.3f}   MAE={row['MAE']:.3f}   R²={row['R2']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cbd5e1"})
        ax.legend(loc="lower right")
        fig.tight_layout()
        path = plot_dir / f"{safe_name(model)}_predicted_vs_nearest_experimental.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    ranked = metrics.sort_values("RMSE", ascending=True)
    height = max(7.0, 0.34 * len(ranked))
    fig, ax = plt.subplots(figsize=(10, height))
    positions = np.arange(len(ranked))
    ax.barh(positions, ranked["RMSE"], color="#2563eb", edgecolor="#1e3a8a", linewidth=0.5)
    ax.set_yticks(positions, ranked["Model"])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE vs nearest experimental value (log10 cm/s)")
    ax.set_title("COSMOS model comparison — lower RMSE is better")
    if len(ranked) > 1:
        display_max = max(1.0, float(ranked["RMSE"].iloc[-2]) * 1.3)
        if float(ranked["RMSE"].max()) > display_max:
            ax.set_xlim(0, display_max)
            ax.text(0.99, 0.01, "Display axis capped; bar-end labels show exact RMSE",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#475569")
    for position, value in zip(positions, ranked["RMSE"]):
        label_x = min(float(value) + 0.01, ax.get_xlim()[1] * 0.94)
        ax.text(label_x, position, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    path = plot_dir / "benchmark_models_rmse_nearest_experimental.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)
    return paths


def run(input_path: Path, output_dir: Path, rdkit_python: str | None = None) -> dict[str, object]:
    paths = ProjectPaths.discover(ROOT)
    source = pd.read_csv(input_path)
    required_columns = {SMILES_COLUMN, TARGET_MEAN, TARGET_VALUES}
    missing = sorted(required_columns - set(source.columns))
    if missing:
        raise ValueError(f"COSMOS input is missing required columns: {missing}")

    dataset = source.copy()
    for source_name, model_name in COSMOS_DESCRIPTOR_ALIASES.items():
        if source_name in dataset.columns and model_name not in dataset.columns:
            dataset[model_name] = pd.to_numeric(dataset[source_name], errors="coerce")

    all_models, load_skips = load_available_models(paths)
    models = [
        model
        for model in all_models
        if model.artifact_path.parent.name == "benchmark"
        and model.artifact_path.parent.parent.name == "reproduction"
    ]
    if not models:
        raise RuntimeError("No compatible benchmark models were loaded.")
    skipped = [message for message in load_skips if message.startswith("Benchmark ")]
    feature_order = []
    for model in models:
        candidates = model.feature_order or tuple(
            column for column in pd.read_csv(paths.final_data / "clean_trial4.csv", nrows=0).columns
            if column not in {"logkpl", "Compound", "SMILES"}
        )
        for feature in candidates:
            if feature not in feature_order:
                feature_order.append(feature)
    request = FeatureRequest(tuple(feature_order))
    features, status, audit, unsupported = prepare_descriptor_frame(
        dataset, [request], SMILES_COLUMN, TARGET_MEAN, paths, rdkit_python=rdkit_python
    )
    valid = status["SMILES_Valid"].astype(bool)

    rows: list[dict[str, object]] = []
    prediction_failures: list[str] = []
    for model in models:
        try:
            native = predict_with_model(model, features.loc[valid])
        except Exception as exc:
            prediction_failures.append(f"{model.name}: {type(exc).__name__}: {exc}")
            continue
        converted = native + PREDICTION_OFFSET
        for position, prediction in zip(np.flatnonzero(valid.to_numpy()), converted):
            record = source.iloc[position]
            nearest = nearest_experimental(record[TARGET_VALUES], float(prediction))
            mean = pd.to_numeric(pd.Series([record[TARGET_MEAN]]), errors="coerce").iloc[0]
            rows.append({
                "Source_Row": int(position), "COSMOS_ID": record.get("COSMOS ID", ""),
                "COSMOS_Name": record.get("COSMOS Name", ""), "SMILES": record[SMILES_COLUMN],
                "Canonical_SMILES": status.iloc[position]["Canonical_SMILES"], "Model": model.name,
                "Artifact_Path": str(model.artifact_path), "Predicted_logKp_cm_per_s": float(prediction),
                "Experimental_logKp_values_cm_per_s": record[TARGET_VALUES],
                "Nearest_Experimental_logKp": nearest, "Experimental_Mean_logKp": mean,
                "Error_vs_Nearest": float(prediction) - nearest, "Absolute_Error_vs_Nearest": abs(float(prediction) - nearest),
                "Error_vs_Mean": float(prediction) - mean, "Absolute_Error_vs_Mean": abs(float(prediction) - mean),
            })
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No model produced predictions.")

    metric_rows = []
    for model, group in predictions.groupby("Model", sort=False):
        nearest_metrics = regression_metrics(group["Nearest_Experimental_logKp"], group["Predicted_logKp_cm_per_s"])
        mean_metrics = regression_metrics(group["Experimental_Mean_logKp"], group["Predicted_logKp_cm_per_s"])
        metric_rows.append({"Model": model, **nearest_metrics, **{f"Mean_{key}": value for key, value in mean_metrics.items()}})
    metrics = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "cosmos_benchmark_model_predictions_long.csv"
    metrics_path = output_dir / "cosmos_benchmark_model_metrics.csv"
    audit_path = output_dir / "cosmos_descriptor_audit.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    audit_models = [
        FeatureRequest(
            model.feature_order or tuple(feature_order),
            display_name=model.name,
        )
        for model in models
    ]
    build_descriptor_audit(audit_models, audit).to_csv(audit_path, index=False)
    plot_paths = make_plots(predictions, metrics, output_dir)
    summary = {
        "input": str(input_path.resolve()), "compound_rows": len(source), "valid_smiles": int(valid.sum()),
        "model_scope": "models/reproduction/benchmark only",
        "models_loaded": len(models), "models_predicted": int(predictions["Model"].nunique()),
        "prediction_rows": len(predictions), "prediction_unit": "log10(cm/s)",
        "native_model_unit": "log10(cm/h)", "unit_offset": PREDICTION_OFFSET,
        "descriptor_aliases_reused": COSMOS_DESCRIPTOR_ALIASES,
        "rdkit_unsupported_training_features_filled_by_cdk": unsupported,
        "model_load_skips": skipped, "prediction_failures": prediction_failures,
        "best_model_by_nearest_rmse": metrics.iloc[0]["Model"], "best_nearest_rmse": float(metrics.iloc[0]["RMSE"]),
        "outputs": {"predictions": str(predictions_path), "metrics": str(metrics_path),
                    "descriptor_audit": str(audit_path), "plots": [str(path) for path in plot_paths]},
    }
    (output_dir / "cosmos_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rdkit-python", default=None)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(run(input_path, output_dir, args.rdkit_python), indent=2))


if __name__ == "__main__":
    main()
