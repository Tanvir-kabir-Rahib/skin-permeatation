from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_permeation.external_validation import (
    calculate_external_validation_metrics,
    calculate_q2_metrics,
    load_prediction_data,
    run_external_validation,
)


def test_q2_metrics_use_standard_external_validation_denominators():
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    y_true = np.array([1.0, 2.0, 4.0])
    y_pred = np.array([1.1, 1.8, 3.7])

    q2 = calculate_q2_metrics(y_true, y_pred, y_train=y_train)

    press = np.sum((y_true - y_pred) ** 2)
    expected_q2_f1 = 1.0 - press / np.sum((y_true - np.mean(y_train)) ** 2)
    expected_q2_f2 = 1.0 - press / np.sum((y_true - np.mean(y_true)) ** 2)
    expected_q2_f3 = 1.0 - (press / len(y_true)) / (np.sum((y_train - np.mean(y_train)) ** 2) / len(y_train))

    assert q2["Q2_F1"] == pytest.approx(expected_q2_f1)
    assert q2["Q2_F2"] == pytest.approx(expected_q2_f2)
    assert q2["Q2_F3"] == pytest.approx(expected_q2_f3)


def test_q2_f1_and_q2_f3_are_nan_without_training_targets():
    y_true = np.array([1.0, 2.0, 4.0])
    y_pred = np.array([1.1, 1.8, 3.7])

    with pytest.warns(RuntimeWarning, match="training-set target values"):
        q2 = calculate_q2_metrics(y_true, y_pred)

    assert np.isnan(q2["Q2_F1"])
    assert np.isfinite(q2["Q2_F2"])
    assert np.isnan(q2["Q2_F3"])


def test_perfect_predictions_pass_core_external_validation_metrics():
    y_true = np.array([-4.0, -3.0, -2.0, -1.0])
    y_pred = y_true.copy()
    y_train = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])

    metrics = calculate_external_validation_metrics(y_true, y_pred, y_train=y_train)

    assert metrics["R2_ext"] == pytest.approx(1.0)
    assert metrics["RMSE_ext"] == pytest.approx(0.0)
    assert metrics["MAE_ext"] == pytest.approx(0.0)
    assert metrics["CCC_ext"] == pytest.approx(1.0)
    assert metrics["r_m^2"] == pytest.approx(1.0)
    assert metrics["Golbraikh-Tropsha criteria"] is True


def test_load_prediction_data_drops_missing_rows(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "Experimental_logKp": [1.0, 2.0, None],
            "Predicted_logKp": [1.1, None, 3.1],
        }
    ).to_csv(csv_path, index=False)

    with pytest.warns(RuntimeWarning, match="Removed 2 row"):
        clean, y_true, y_pred = load_prediction_data(csv_path, "Experimental_logKp", "Predicted_logKp")

    assert len(clean) == 1
    assert y_true.tolist() == [1.0]
    assert y_pred.tolist() == [1.1]


def test_run_external_validation_writes_requested_outputs(tmp_path):
    predictions_path = tmp_path / "external_test_predictions.csv"
    train_path = tmp_path / "train_data.csv"
    output_dir = tmp_path / "outputs" / "external_validation"

    pd.DataFrame(
        {
            "Experimental_logKp": [-4.0, -3.0, -2.0, -1.0],
            "Predicted_logKp": [-4.1, -2.9, -2.2, -0.8],
        }
    ).to_csv(predictions_path, index=False)
    pd.DataFrame({"Experimental_logKp": [-5.0, -4.0, -3.0, -2.0, -1.0]}).to_csv(train_path, index=False)

    summary = run_external_validation(
        csv_path=predictions_path,
        actual_col="Experimental_logKp",
        pred_col="Predicted_logKp",
        train_csv_path=train_path,
        train_target_col="Experimental_logKp",
        output_dir=output_dir,
    )

    assert list(summary.columns) == ["Metric", "Value", "Acceptance Criterion", "Result"]
    assert (output_dir / "external_validation_metrics.csv").exists()
    assert (output_dir / "external_validation_metrics.xlsx").exists()
    assert (output_dir / "journal_ready_external_validation_table.csv").exists()
    assert (output_dir / "journal_ready_external_validation_table.xlsx").exists()
    assert (output_dir / "experimental_vs_predicted_logKp.png").exists()
