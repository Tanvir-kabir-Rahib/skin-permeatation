from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_permeation.huskin import (
    BenchmarkModel,
    calculate_missing_rdkit_descriptors,
    ensure_target_excluded,
    normalize_feature_name,
    prepare_model_features,
    resolve_prediction_unit,
)


class OrderedDummyEstimator:
    feature_names_in_ = np.asarray(["A", "B"], dtype=object)

    def predict(self, frame):
        return np.asarray(frame["A"] + frame["B"], dtype=float)


def _dummy_model(features=("A", "B")):
    return BenchmarkModel(
        display_name="Dummy",
        output_name="Dummy",
        artifact_path=ROOT / "dummy.joblib",
        estimator=OrderedDummyEstimator(),
        required_features=tuple(features),
        serialization_format="joblib",
        preprocessing_summary="dummy",
    )


def test_descriptor_name_normalization_handles_harmless_differences():
    assert normalize_feature_name("  Topo PSA ") == normalize_feature_name("topo-psa")
    assert normalize_feature_name("XLogP") != normalize_feature_name("LogP")


def test_prepare_model_features_preserves_exact_model_order():
    frame = pd.DataFrame({"B": [2.0], "A": [1.0], "unused": [9.0]})
    ordered = prepare_model_features(frame, _dummy_model())
    assert ordered.columns.tolist() == ["A", "B"]
    assert ordered.to_numpy().tolist() == [[1.0, 2.0]]


def test_target_column_is_rejected_from_model_features():
    with pytest.raises(Exception, match="Target leakage"):
        ensure_target_excluded([_dummy_model(("A", "logkp (cm/s)"))], "logkp (cm/s)")


def test_prediction_units_are_converted_from_cm_per_hour_to_cm_per_second():
    unit, offset = resolve_prediction_unit("logkp (cm/s)")
    assert unit == "cm/s"
    assert offset == pytest.approx(-np.log10(3600.0))


def test_rdkit_handles_invalid_smiles_and_generates_supported_descriptors():
    result = calculate_missing_rdkit_descriptors(
        pd.Series(["c1ccccc1", "not-a-smiles", "c1ccccc1"]),
        ["nAromRings", "Zagreb"],
    )
    assert bool(result.loc[0, "valid"])
    assert result.loc[0, "nAromRings"] == pytest.approx(1.0)
    assert np.isfinite(float(result.loc[0, "Zagreb"]))
    assert not bool(result.loc[1, "valid"])
    assert "parse" in str(result.loc[1, "error"]).casefold()
    assert bool(result.loc[2, "valid"])
    assert result.loc[2, "canonical_smiles"] == result.loc[0, "canonical_smiles"]
    assert result.loc[2, "Zagreb"] == result.loc[0, "Zagreb"]


def test_prediction_output_shape_for_saved_benchmark_model():
    joblib = pytest.importorskip("joblib")
    artifact = ROOT / "models" / "reproduction" / "benchmark" / "RF.joblib"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")
        estimator = joblib.load(artifact)
    features = tuple(str(value) for value in estimator.feature_names_in_)
    training = pd.read_csv(ROOT / "data" / "final" / "clean_trial4.csv")
    predicted = estimator.predict(training.loc[:1, list(features)])
    assert np.asarray(predicted).shape == (2,)
