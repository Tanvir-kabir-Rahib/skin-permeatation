from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_permeation.benchmark_external_validation import (
    THRESHOLDED_METRICS,
    load_benchmark_inventory,
    validate_all_benchmark_models,
)


def _write_test_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = pd.DataFrame(
        {
            "Compound": [f"compound_{index}" for index in range(30)],
            "SMILES": [f"smiles_{index // 2}" for index in range(30)],
            "logkpl": np.linspace(-4.5, -0.5, 30),
            "descriptor_1": np.linspace(0.0, 1.0, 30),
            "descriptor_2": np.linspace(1.0, 3.0, 30) ** 2,
        }
    )
    dataset_path = tmp_path / "data.csv"
    dataset.to_csv(dataset_path, index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for name, estimator in {
        "Linear": LinearRegression(),
        "Dummy": DummyRegressor(strategy="mean"),
    }.items():
        estimator.fit(dataset[["descriptor_1", "descriptor_2"]], dataset["logkpl"])
        joblib.dump(estimator, models_dir / f"{name}.joblib")

    metrics_path = tmp_path / "benchmark_metrics.csv"
    pd.DataFrame(
        {
            "model": ["Linear", "Dummy"],
            "artifact_path": [models_dir / "Linear.joblib", models_dir / "Dummy.joblib"],
        }
    ).to_csv(metrics_path, index=False)
    return dataset_path, metrics_path, models_dir


def test_inventory_uses_each_benchmark_artifact_once(tmp_path):
    _, metrics_path, models_dir = _write_test_artifacts(tmp_path)
    joblib.dump(LinearRegression(), models_dir / "best_benchmark_model.joblib")

    inventory = load_benchmark_inventory(metrics_path, models_dir)

    assert inventory["Model"].tolist() == ["Linear", "Dummy"]
    assert not inventory["Source Artifact"].str.contains("best_benchmark_model").any()


def test_all_model_validation_uses_one_holdout_and_writes_auditable_tables(tmp_path):
    dataset_path, metrics_path, models_dir = _write_test_artifacts(tmp_path)
    output_dir = tmp_path / "outputs"

    result = validate_all_benchmark_models(
        dataset_csv_path=dataset_path,
        benchmark_metrics_path=metrics_path,
        models_dir=models_dir,
        output_dir=output_dir,
        model_names=["Linear", "Dummy"],
        save_fitted_models=False,
        make_plots=False,
    )

    assert result.summary["Model"].nunique() == 2
    assert result.metrics_long.groupby("Model").size().eq(18).all()
    validation_rows = result.predictions_long.groupby("Model")["Validation Row"].apply(tuple)
    assert validation_rows.nunique() == 1
    assert result.protocol["models"]["external_labels_used_for_training_or_tuning"] is False
    assert result.protocol["acceptance_summary"]["criteria_per_model"] == len(THRESHOLDED_METRICS)
    assert (output_dir / "benchmark_external_validation_summary.csv").exists()
    assert (output_dir / "benchmark_external_validation_metrics_long.csv").exists()
    assert (output_dir / "benchmark_external_validation_predictions_long.csv").exists()
    assert (output_dir / "benchmark_external_validation_protocol.json").exists()
