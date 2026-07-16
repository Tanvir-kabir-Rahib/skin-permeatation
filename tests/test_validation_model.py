from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_permeation.validation_model import (
    build_candidate_models,
    fit_cross_validated_ensemble,
    make_cv_splitter,
    predict_selected_model,
    prepare_modeling_data,
    resolve_validation_protocol,
    split_external_holdout,
)


def test_grouped_external_holdout_has_no_smiles_overlap():
    frame = pd.DataFrame(
        {
            "SMILES": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "Compound": [f"compound_{index}" for index in range(8)],
            "logkpl": np.linspace(-4.0, -1.0, 8),
            "descriptor_1": np.arange(8, dtype=float),
            "descriptor_2": np.arange(8, dtype=float) ** 2,
        }
    )
    features, y, identifiers = prepare_modeling_data(frame, "logkpl")
    train_indices, test_indices = split_external_holdout(
        features,
        y,
        identifiers,
        split_strategy="grouped",
        test_size=0.25,
        random_state=42,
    )

    train_smiles = set(identifiers.iloc[train_indices]["SMILES"])
    test_smiles = set(identifiers.iloc[test_indices]["SMILES"])
    assert train_smiles.isdisjoint(test_smiles)


def test_prepare_modeling_data_excludes_identifiers_and_target():
    frame = pd.DataFrame(
        {
            "SMILES": ["A", "B"],
            "Compound": ["one", "two"],
            "logkpl": [-2.0, -1.0],
            "numeric_descriptor": [1.0, 2.0],
            "text_descriptor": ["x", "y"],
        }
    )
    features, y, identifiers = prepare_modeling_data(frame, "logkpl")

    assert list(features.columns) == ["numeric_descriptor"]
    assert y.tolist() == [-2.0, -1.0]
    assert list(identifiers.columns) == ["Compound", "SMILES"]


def test_fast_cross_validated_ensemble_produces_finite_predictions():
    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(size=(36, 6)), columns=[f"x_{index}" for index in range(6)])
    y = 1.5 * features["x_0"].to_numpy() - 0.8 * features["x_1"].to_numpy() + rng.normal(0, 0.1, 36)
    identifiers = pd.DataFrame({"SMILES": [f"molecule_{index}" for index in range(36)]})
    train_indices = np.arange(30)
    cv, groups = make_cv_splitter(
        "random",
        identifiers,
        train_indices,
        cv_folds=3,
        random_state=42,
    )

    selected_name, fitted_models, meta_model, scores = fit_cross_validated_ensemble(
        features.iloc[train_indices],
        y[train_indices],
        cv=cv,
        groups=groups,
        random_state=42,
        fast=True,
    )
    predictions = predict_selected_model(
        selected_name,
        fitted_models,
        meta_model,
        features.iloc[30:],
    )

    assert predictions.shape == (6,)
    assert np.isfinite(predictions).all()
    assert selected_name in scores


def test_candidate_search_includes_training_only_calibrated_scores():
    rng = np.random.default_rng(7)
    features = pd.DataFrame(rng.normal(size=(30, 4)), columns=[f"x_{index}" for index in range(4)])
    y = 2.0 * features["x_0"].to_numpy() + rng.normal(0, 0.2, 30)
    identifiers = pd.DataFrame({"SMILES": [f"molecule_{index}" for index in range(30)]})
    train_indices = np.arange(24)
    cv, groups = make_cv_splitter("random", identifiers, train_indices, 3, 42)

    _, _, _, scores = fit_cross_validated_ensemble(
        features.iloc[train_indices],
        y[train_indices],
        cv=cv,
        groups=groups,
        random_state=42,
        fast=True,
    )

    for model_name in build_candidate_models(random_state=42, fast=True):
        assert model_name in scores
        assert f"{model_name}_Calibrated" in scores
    assert "OOF_Ridge_Ensemble_Calibrated" in scores


def test_validation_protocol_defaults_match_strict_and_paper_workflows():
    assert resolve_validation_protocol("strict-grouped", None, None, None) == ("grouped", 0.20, 42)
    assert resolve_validation_protocol("paper-reproduction", None, None, None) == ("random", 0.15, 1)
