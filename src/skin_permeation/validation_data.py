from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


TARGET_COLUMN_CANDIDATES = (
    "Experimental_logKp",
    "logkpl",
    "logKp",
    "LogKp",
    "log_kp",
    "Log_Kp",
    "target",
    "y",
)
ACTUAL_COLUMN_CANDIDATES = (
    "Experimental_logKp",
    "actual_logkpl",
    "Actual",
    "actual",
    "Observed",
    "observed",
    "y_true",
    "test_actual",
    "actual_logKp",
)
PREDICTED_COLUMN_CANDIDATES = (
    "Predicted_logKp",
    "predicted_logkpl",
    "Predicted",
    "predicted",
    "Prediction",
    "prediction",
    "y_pred",
    "test_predicted",
    "predicted_logKp",
)
MATCH_COLUMN_CANDIDATES = (
    "row_index",
    "original_index",
    "index",
    "SMILES",
    "smiles",
    "Compound",
    "compound",
    "Compound_ID",
    "compound_id",
    "ID",
    "id",
)
GROUP_COLUMN_CANDIDATES = ("SMILES", "smiles")


def normalize_column_name(name: str) -> str:
    """Normalize a column name for case-insensitive, punctuation-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def resolve_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: tuple[str, ...],
    description: str,
) -> str:
    """Resolve a requested or commonly named column and raise a helpful error if none is found."""
    normalized_columns = {normalize_column_name(column): column for column in frame.columns}
    if requested is not None:
        match = normalized_columns.get(normalize_column_name(requested))
        if match is not None:
            return match
        raise ValueError(
            f"Could not find requested {description} column '{requested}'. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )

    for candidate in candidates:
        match = normalized_columns.get(normalize_column_name(candidate))
        if match is not None:
            return match

    if description == "training target":
        logkp_like_columns = [
            column
            for column in frame.columns
            if "logkp" in normalize_column_name(column)
        ]
        if len(logkp_like_columns) == 1:
            return logkp_like_columns[0]
    elif description == "actual response":
        actual_logkp_columns = [
            column
            for column in frame.columns
            if "logkp" in normalize_column_name(column)
            and any(
                marker in normalize_column_name(column)
                for marker in ("actual", "experimental", "observed", "true")
            )
        ]
        if len(actual_logkp_columns) == 1:
            return actual_logkp_columns[0]
    elif description == "predicted response":
        predicted_logkp_columns = [
            column
            for column in frame.columns
            if "logkp" in normalize_column_name(column)
            and any(
                marker in normalize_column_name(column)
                for marker in ("predicted", "prediction", "estimated", "pred")
            )
        ]
        if len(predicted_logkp_columns) == 1:
            return predicted_logkp_columns[0]

    raise ValueError(
        f"Could not automatically identify the {description} column. "
        f"Pass it explicitly. Available columns: {', '.join(map(str, frame.columns))}"
    )


def resolve_shared_match_column(
    full_data: pd.DataFrame,
    external_predictions: pd.DataFrame,
    requested: str | None = None,
) -> tuple[str, str] | None:
    """Find identifier columns shared by the complete dataset and external prediction artifact."""
    if requested is not None:
        full_column = resolve_column(full_data, requested, (requested,), "matching identifier")
        external_column = resolve_column(external_predictions, requested, (requested,), "matching identifier")
        return full_column, external_column

    full_columns = {normalize_column_name(column): column for column in full_data.columns}
    external_columns = {normalize_column_name(column): column for column in external_predictions.columns}
    for candidate in MATCH_COLUMN_CANDIDATES:
        normalized = normalize_column_name(candidate)
        if normalized in full_columns and normalized in external_columns:
            return full_columns[normalized], external_columns[normalized]
    return None


def load_csv(csv_path: str | Path, description: str) -> pd.DataFrame:
    """Load a CSV file and provide a clear error when the file is absent or empty."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"{description} CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{description} CSV contains no rows: {path}")
    return frame


def build_external_predictions(
    predictions: pd.DataFrame,
    actual_col: str | None = None,
    pred_col: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Normalize a project prediction artifact to the external validation input schema."""
    resolved_actual = resolve_column(predictions, actual_col, ACTUAL_COLUMN_CANDIDATES, "actual response")
    resolved_predicted = resolve_column(predictions, pred_col, PREDICTED_COLUMN_CANDIDATES, "predicted response")
    valid = predictions.dropna(subset=[resolved_actual, resolved_predicted]).copy()
    if valid.empty:
        raise ValueError("No complete actual/predicted pairs remain in the prediction artifact.")

    output = pd.DataFrame(
        {
            "Experimental_logKp": pd.to_numeric(valid[resolved_actual], errors="raise"),
            "Predicted_logKp": pd.to_numeric(valid[resolved_predicted], errors="raise"),
        }
    )
    identifier_columns = [
        column
        for column in MATCH_COLUMN_CANDIDATES
        if column in valid.columns and column not in {resolved_actual, resolved_predicted}
    ]
    if identifier_columns:
        output = pd.concat([valid[identifier_columns].reset_index(drop=True), output.reset_index(drop=True)], axis=1)
    return output, resolved_actual, resolved_predicted


def training_rows_from_identifier_match(
    full_data: pd.DataFrame,
    external_predictions: pd.DataFrame,
    match_columns: tuple[str, str],
) -> pd.DataFrame:
    """Remove external-test identifiers from the complete dataset to recover training rows."""
    full_match_col, external_match_col = match_columns
    external_ids = set(external_predictions[external_match_col].dropna().astype(str))
    if not external_ids:
        raise ValueError(f"External prediction identifier column '{external_match_col}' contains no usable values.")
    training_rows = full_data.loc[~full_data[full_match_col].astype(str).isin(external_ids)].copy()
    if training_rows.empty:
        raise ValueError("Identifier matching removed every row; check that the selected match column is correct.")
    return training_rows


def training_rows_from_reproduced_split(
    full_data: pd.DataFrame,
    target_col: str,
    group_col: str | None,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    """Reproduce a grouped or ordinary holdout split and return its training rows."""
    valid = full_data.dropna(subset=[target_col]).copy()
    if len(valid) < 3:
        raise ValueError("At least three rows with valid target values are required to reproduce a split.")

    if group_col is not None:
        grouped = valid.dropna(subset=[group_col]).copy()
        if grouped[group_col].nunique() < 2:
            raise ValueError(f"Group column '{group_col}' must contain at least two distinct groups.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_positions, _ = next(splitter.split(grouped, groups=grouped[group_col].astype(str)))
        return grouped.iloc[train_positions].copy()

    train_rows, _ = train_test_split(valid, test_size=test_size, random_state=random_state)
    return train_rows.copy()


def resolve_optional_group_column(full_data: pd.DataFrame, requested: str | None) -> str | None:
    """Resolve the molecular grouping column, returning None when no grouping column is available."""
    if requested is not None:
        return resolve_column(full_data, requested, (requested,), "group")
    normalized_columns = {normalize_column_name(column): column for column in full_data.columns}
    for candidate in GROUP_COLUMN_CANDIDATES:
        match = normalized_columns.get(normalize_column_name(candidate))
        if match is not None:
            return match
    return None


def generate_validation_input_files(
    dataset_csv_path: str | Path = "data/final/clean_trial4.csv",
    predictions_csv_path: str | Path = "reports/tables/improved_test_errors.csv",
    target_col: str | None = None,
    actual_col: str | None = None,
    pred_col: str | None = None,
    match_col: str | None = None,
    group_col: str | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
    output_dir: str | Path = ".",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train_data.csv and external_test_predictions.csv from project artifacts.

    The external prediction file is normalized from a saved test-prediction artifact. Training
    rows are recovered by removing test identifiers from the complete dataset when a shared
    identifier is available. Otherwise, the function reproduces a grouped split when a SMILES
    column exists, or an ordinary random split as the final fallback.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    full_data = load_csv(dataset_csv_path, "Complete modeling dataset")
    predictions = load_csv(predictions_csv_path, "External prediction artifact")
    resolved_target = resolve_column(full_data, target_col, TARGET_COLUMN_CANDIDATES, "training target")
    external_output, _, _ = build_external_predictions(predictions, actual_col=actual_col, pred_col=pred_col)

    match_columns = resolve_shared_match_column(full_data, predictions, requested=match_col)
    if match_columns is not None:
        training_rows = training_rows_from_identifier_match(full_data, predictions, match_columns)
        split_note = f"matched using '{match_columns[0]}'"
    else:
        resolved_group = resolve_optional_group_column(full_data, group_col)
        warnings.warn(
            "No shared test-row identifier was found. Reproducing the train/test split; make sure "
            "test_size and random_state match the model run.",
            RuntimeWarning,
            stacklevel=2,
        )
        training_rows = training_rows_from_reproduced_split(
            full_data,
            target_col=resolved_target,
            group_col=resolved_group,
            test_size=test_size,
            random_state=random_state,
        )
        split_note = (
            f"reproduced with group column '{resolved_group}'"
            if resolved_group is not None
            else "reproduced with an ordinary random split"
        )

    training_target = pd.to_numeric(training_rows[resolved_target], errors="coerce").dropna()
    if training_target.empty:
        raise ValueError(f"No numeric training targets were found in '{resolved_target}'.")
    train_output = pd.DataFrame({"Experimental_logKp": training_target.to_numpy()})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_data.csv"
    external_path = output_dir / "external_test_predictions.csv"
    train_output.to_csv(train_path, index=False)
    external_output.to_csv(external_path, index=False)

    print("Validation input files generated successfully")
    print(f"Training targets: {len(train_output)} rows -> {train_path.resolve()}")
    print(f"External predictions: {len(external_output)} rows -> {external_path.resolve()}")
    print(f"Training rows were {split_note}.")
    return train_output, external_output
