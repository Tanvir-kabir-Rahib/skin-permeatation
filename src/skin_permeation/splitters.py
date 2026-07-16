from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dependencies import require_module

sk_model_selection = require_module("sklearn.model_selection", "Install scikit-learn to use splitting utilities.")

GroupShuffleSplit = sk_model_selection.GroupShuffleSplit
GroupKFold = sk_model_selection.GroupKFold
KFold = sk_model_selection.KFold
RepeatedKFold = sk_model_selection.RepeatedKFold
train_test_split = sk_model_selection.train_test_split


@dataclass(frozen=True)
class SplitResult:
    train_index: np.ndarray
    test_index: np.ndarray


def make_groups(df: pd.DataFrame, molecule_column: str = "SMILES") -> pd.Series:
    if molecule_column not in df.columns:
        raise KeyError(f"Grouping column '{molecule_column}' not present in dataframe.")
    return df[molecule_column].astype(str)


def naive_random_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> SplitResult:
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    return SplitResult(np.asarray(train_idx), np.asarray(test_idx))


def grouped_split(
    df: pd.DataFrame,
    groups: pd.Series,
    test_size: float,
    random_state: int,
) -> SplitResult:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return SplitResult(np.asarray(train_idx), np.asarray(test_idx))
