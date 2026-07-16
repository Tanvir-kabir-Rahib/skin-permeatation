from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from skin_permeation.data_audit import profile_dataset


def test_profile_dataset_identifies_targets_and_ids():
    frame = pd.DataFrame(
        {
            "Compound": ["A", "B"],
            "SMILES": ["CC", "CO"],
            "logkpl": [-1.0, -2.0],
            "Texpi": [305, 310],
        }
    )
    profile = profile_dataset("toy", frame)
    assert profile.target_columns == ["logkpl", "Texpi"]
    assert set(profile.identifier_columns) == {"Compound", "SMILES"}
