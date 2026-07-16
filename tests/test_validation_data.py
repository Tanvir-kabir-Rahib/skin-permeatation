from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_permeation.validation_data import generate_validation_input_files


def test_generate_validation_files_by_matching_smiles(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    predictions_path = tmp_path / "predictions.csv"

    pd.DataFrame(
        {
            "SMILES": ["CC", "CCC", "CCCC", "CCO"],
            "logKp": [-1.0, -2.0, -3.0, -4.0],
            "descriptor": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(dataset_path, index=False)
    pd.DataFrame(
        {
            "SMILES": ["CCC", "CCO"],
            "Actual": [-2.0, -4.0],
            "Predicted": [-2.2, -3.8],
        }
    ).to_csv(predictions_path, index=False)

    train_data, external_data = generate_validation_input_files(
        dataset_csv_path=dataset_path,
        predictions_csv_path=predictions_path,
        output_dir=tmp_path,
    )

    assert train_data["Experimental_logKp"].tolist() == [-1.0, -3.0]
    assert external_data["Experimental_logKp"].tolist() == [-2.0, -4.0]
    assert external_data["Predicted_logKp"].tolist() == [-2.2, -3.8]
    assert (tmp_path / "train_data.csv").exists()
    assert (tmp_path / "external_test_predictions.csv").exists()


def test_generate_validation_files_detects_project_logkpl_target(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    predictions_path = tmp_path / "predictions.csv"

    pd.DataFrame(
        {
            "SMILES": ["CC", "CCC", "CCCC"],
            "logkpl": [-1.0, -2.0, -3.0],
        }
    ).to_csv(dataset_path, index=False)
    pd.DataFrame(
        {
            "SMILES": ["CCC"],
            "Actual": [-2.0],
            "Predicted": [-2.1],
        }
    ).to_csv(predictions_path, index=False)

    train_data, _ = generate_validation_input_files(
        dataset_csv_path=dataset_path,
        predictions_csv_path=predictions_path,
        output_dir=tmp_path,
    )

    assert train_data["Experimental_logKp"].tolist() == [-1.0, -3.0]


def test_generate_validation_files_detects_project_prediction_columns(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    predictions_path = tmp_path / "predictions.csv"

    pd.DataFrame(
        {
            "SMILES": ["CC", "CCC", "CCCC"],
            "logkpl": [-1.0, -2.0, -3.0],
        }
    ).to_csv(dataset_path, index=False)
    pd.DataFrame(
        {
            "SMILES": ["CCC"],
            "actual_logkpl": [-2.0],
            "predicted_logkpl": [-2.1],
            "absolute_error": [0.1],
        }
    ).to_csv(predictions_path, index=False)

    _, external_data = generate_validation_input_files(
        dataset_csv_path=dataset_path,
        predictions_csv_path=predictions_path,
        output_dir=tmp_path,
    )

    assert external_data["Experimental_logKp"].tolist() == [-2.0]
    assert external_data["Predicted_logKp"].tolist() == [-2.1]


def test_generate_validation_files_reproduces_split_without_identifier(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    predictions_path = tmp_path / "predictions.csv"

    pd.DataFrame(
        {
            "logKp": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "descriptor": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).to_csv(dataset_path, index=False)
    pd.DataFrame(
        {
            "y_true": [-2.0, -5.0],
            "y_pred": [-2.1, -4.7],
        }
    ).to_csv(predictions_path, index=False)

    with pytest.warns(RuntimeWarning, match="No shared test-row identifier"):
        train_data, external_data = generate_validation_input_files(
            dataset_csv_path=dataset_path,
            predictions_csv_path=predictions_path,
            test_size=0.40,
            random_state=1,
            output_dir=tmp_path,
        )

    assert len(train_data) == 3
    assert list(external_data.columns) == ["Experimental_logKp", "Predicted_logKp"]
