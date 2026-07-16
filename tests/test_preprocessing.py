from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from skin_permeation.modeling.transforms import CorrelationFilter
from skin_permeation.preprocessing import correlated_feature_drops, reconstruct_clean_trial4


def test_correlated_feature_drops_marks_one_side_of_perfectly_correlated_pair():
    frame = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [2, 4, 6, 8],
            "feature_c": [4, 1, 2, 3],
        }
    )
    dropped = correlated_feature_drops(frame, threshold=0.95)
    assert "feature_b" in dropped


def test_reconstruct_clean_trial4_removes_duplicates_and_manual_columns():
    trial4 = pd.DataFrame(
        {
            "logkpl": [-1.0, -1.0],
            "Compound": ["A", "A"],
            "SMILES": ["CC", "CC"],
            "Texpi": [310, 310],
            "khs.ssssB": [0.0, 0.0],
            "SCH-3": [0.0, 0.0],
            "descriptor_x": [1.0, 1.0],
        }
    )
    result = reconstruct_clean_trial4(trial4)
    assert result.dropped_duplicate_rows == 1
    assert "khs.ssssB" not in result.cleaned_trial4.columns
    assert "SCH-3" not in result.cleaned_trial4.columns


def test_correlation_filter_keeps_dataframe_columns_consistent():
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],
            "c": [1, 0, 1, 0],
        }
    )
    transformed = CorrelationFilter(threshold=0.95).fit_transform(frame)
    assert list(transformed.columns) == ["a", "c"]
