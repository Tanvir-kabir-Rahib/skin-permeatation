from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pytest

from skin_permeation.data_loading import load_bundle
from skin_permeation.formulas import calculate_formula_logkp
from skin_permeation.paths import ProjectPaths
from skin_permeation.prediction import (
    align_descriptor_frame_to_feature_columns,
    build_inference_fill_values,
    feature_columns_from_training_frame,
    load_available_models,
    predict_with_model,
)


def test_formula_logkp_matches_potts_guy_reference():
    assert calculate_formula_logkp(log_p=2.0, molecular_weight=100.0) == pytest.approx(-1.91)


def test_descriptor_alignment_matches_training_feature_order_for_existing_row():
    paths = ProjectPaths.discover(ROOT)
    bundle = load_bundle(paths)
    feature_columns = feature_columns_from_training_frame(bundle.clean_trial4)
    fill_values = build_inference_fill_values(bundle)

    descriptor_frame = bundle.trial4.head(1).copy()
    aligned = align_descriptor_frame_to_feature_columns(
        descriptor_frame=descriptor_frame,
        feature_columns=feature_columns,
        fill_values=fill_values,
        texpi=float(descriptor_frame.iloc[0]["Texpi"]),
    )

    expected = descriptor_frame.loc[:, feature_columns].apply(lambda column: column.astype(float))
    assert list(aligned.columns) == feature_columns
    assert not aligned.isna().any().any()
    assert np.allclose(aligned.to_numpy(dtype=float), expected.to_numpy(dtype=float))


def test_primary_model_predicts_on_existing_training_row():
    paths = ProjectPaths.discover(ROOT)
    bundle = load_bundle(paths)
    feature_columns = feature_columns_from_training_frame(bundle.clean_trial4)
    loaded_models, _ = load_available_models(paths)
    primary_model = next(model for model in loaded_models if model.is_primary)

    prediction = predict_with_model(primary_model, bundle.clean_trial4[feature_columns].head(1))
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])
