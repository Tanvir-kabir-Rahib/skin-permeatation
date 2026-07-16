from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PAPER_MANUAL_DROP_COLUMNS: list[str] = [
    "khs.ssssB",
    "khs.ssssBe",
    "SCH-4",
    "C1SP1",
    "khs.tsC",
    "khs.sssNH",
    "khs.sI",
    "SCH-3",
]

IMPUTATION_STRATEGIES: dict[str, str] = {
    "topoShape": "mean",
    "HybRatio": "median",
    "JPLogP": "median",
    "Kier3": "median",
}


@dataclass(frozen=True)
class ReconstructionArtifacts:
    cleaned_trial4: pd.DataFrame
    dropped_duplicate_rows: int
    dropped_correlated_features: list[str]
    dropped_manual_features: list[str]


def remove_water_rows(df: pd.DataFrame, start: int = 445, stop: int = 476) -> pd.DataFrame:
    LOGGER.info("Removing notebook-indicated water rows in [%s, %s).", start, stop)
    return df.drop(index=list(range(start, stop))).reset_index(drop=True)


def compute_imputation_values(
    df: pd.DataFrame,
    strategies: dict[str, str] | None = None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for column, strategy in (strategies or IMPUTATION_STRATEGIES).items():
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        if strategy == "mean":
            values[column] = float(series.mean())
        elif strategy == "median":
            values[column] = float(series.median())
        else:
            raise ValueError(f"Unsupported imputation strategy {strategy!r} for column {column!r}.")
    return values


def apply_imputation_values(df: pd.DataFrame, values: dict[str, float]) -> pd.DataFrame:
    output = df.copy()
    for column, value in values.items():
        if column in output.columns:
            output[column] = output[column].fillna(value)
    return output


def impute_trial4_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return apply_imputation_values(df, compute_imputation_values(df))


def correlated_feature_drops(
    frame: pd.DataFrame,
    threshold: float = 0.95,
    additional_drop_columns: Sequence[str] | None = None,
) -> list[str]:
    numeric = frame.drop(columns=list(additional_drop_columns or []), errors="ignore")
    corr = numeric.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: list[str] = []
    for column in upper.columns:
        if any(upper[column] >= threshold):
            to_drop.append(column)
    return sorted(set(to_drop))


def reconstruct_clean_trial4(
    trial4: pd.DataFrame,
    correlation_threshold: float = 0.95,
) -> ReconstructionArtifacts:
    deduplicated = trial4.drop_duplicates().reset_index(drop=True)
    modeling_frame = deduplicated.drop(columns=["SMILES", "Compound"])
    dropped_correlated = correlated_feature_drops(
        modeling_frame,
        threshold=correlation_threshold,
        additional_drop_columns=PAPER_MANUAL_DROP_COLUMNS,
    )
    cleaned = deduplicated.drop(columns=[*PAPER_MANUAL_DROP_COLUMNS, *dropped_correlated], errors="ignore")
    return ReconstructionArtifacts(
        cleaned_trial4=cleaned,
        dropped_duplicate_rows=int(len(trial4) - len(deduplicated)),
        dropped_correlated_features=dropped_correlated,
        dropped_manual_features=list(PAPER_MANUAL_DROP_COLUMNS),
    )
