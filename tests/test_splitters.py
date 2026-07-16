from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from skin_permeation.splitters import grouped_split


def test_grouped_split_keeps_groups_disjoint():
    frame = pd.DataFrame({"SMILES": ["A", "A", "B", "B", "C", "C"], "value": range(6)})
    split = grouped_split(frame, frame["SMILES"], test_size=0.33, random_state=1)
    train_groups = set(frame.iloc[split.train_index]["SMILES"])
    test_groups = set(frame.iloc[split.test_index]["SMILES"])
    assert train_groups.isdisjoint(test_groups)
