from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVR

from .external_validation import calculate_external_validation_metrics, run_external_validation
from .validation_data import TARGET_COLUMN_CANDIDATES, resolve_column


IDENTIFIER_COLUMNS = ("Compound", "SMILES", "compound", "smiles", "ID", "id")
VALIDATION_PROTOCOLS = ("strict-grouped", "paper-reproduction")
MIN_CALIBRATION_R2_GAIN = 0.01
MAX_CALIBRATION_AGREEMENT_LOSS = 0.01
MAX_CALIBRATION_DELTA_RM2_INCREASE = 0.02


@dataclass
class ValidationTrainingResult:
    """Artifacts and scores produced by validation-ready model training."""

    selected_model: str
    cross_validated_scores: dict[str, float]
    train_data: pd.DataFrame
    external_predictions: pd.DataFrame
    summary_table: pd.DataFrame


def resolve_validation_protocol(
    protocol: str,
    split_strategy: str | None,
    test_size: float | None,
    random_state: int | None,
) -> tuple[str, float, int]:
    """Resolve strict or paper-reproduction split settings.

    The strict protocol keeps distinct SMILES groups apart. The paper-reproduction protocol
    mirrors the project's original 85/15 random split with random_state=1; it is useful for
    reproducing reported performance but is not equivalent to unseen-scaffold validation.
    """
    if protocol not in VALIDATION_PROTOCOLS:
        raise ValueError(f"protocol must be one of: {', '.join(VALIDATION_PROTOCOLS)}")

    defaults = {
        "strict-grouped": ("grouped", 0.20, 42),
        "paper-reproduction": ("random", 0.15, 1),
    }
    default_strategy, default_test_size, default_random_state = defaults[protocol]
    return (
        split_strategy or default_strategy,
        default_test_size if test_size is None else test_size,
        default_random_state if random_state is None else random_state,
    )


def load_modeling_dataset(csv_path: str | Path, target_col: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load a modeling CSV and resolve its experimental target column."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Modeling dataset not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Modeling dataset contains no rows: {path}")
    resolved_target = resolve_column(frame, target_col, TARGET_COLUMN_CANDIDATES, "training target")
    return frame, resolved_target


def prepare_modeling_data(
    frame: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Extract numeric descriptor features, target values, and available identifiers."""
    valid = frame.dropna(subset=[target_col]).copy()
    numeric_target = pd.to_numeric(valid[target_col], errors="coerce")
    valid = valid.loc[numeric_target.notna()].copy()
    y = numeric_target.loc[numeric_target.notna()].to_numpy(dtype=float)

    identifier_columns = [column for column in IDENTIFIER_COLUMNS if column in valid.columns]
    identifiers = valid[identifier_columns].copy() if identifier_columns else pd.DataFrame(index=valid.index)

    excluded = {target_col, *identifier_columns}
    features = valid.drop(columns=[column for column in excluded if column in valid.columns]).copy()
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.loc[:, features.notna().any(axis=0)]
    if features.empty:
        raise ValueError("No numeric descriptor columns are available after excluding identifiers and target.")
    return features, y, identifiers


def split_external_holdout(
    features: pd.DataFrame,
    y: np.ndarray,
    identifiers: pd.DataFrame,
    split_strategy: str,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a grouped unseen-compound holdout or a paper-style random holdout."""
    indices = np.arange(len(features))
    if split_strategy == "grouped":
        smiles_column = next((column for column in ("SMILES", "smiles") if column in identifiers.columns), None)
        if smiles_column is None:
            raise ValueError("Grouped validation requires a SMILES column in the modeling dataset.")
        groups = identifiers[smiles_column].fillna("__missing_smiles__").astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        return next(splitter.split(indices, y, groups=groups))
    if split_strategy == "random":
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
        )
        return np.asarray(train_indices), np.asarray(test_indices)
    raise ValueError("split_strategy must be either 'grouped' or 'random'.")


def build_candidate_models(random_state: int, fast: bool = False) -> dict[str, object]:
    """Build diverse nonlinear regression candidates for cross-validated model selection."""
    tree_count = 80 if fast else 700
    boosting_count = 80 if fast else 500

    def pipeline(model: object) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )

    candidates = {
        "ExtraTrees": pipeline(
            ExtraTreesRegressor(
                n_estimators=tree_count,
                max_features=0.80,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "RandomForest": pipeline(
            RandomForestRegressor(
                n_estimators=tree_count,
                max_features=0.70,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "GradientBoosting": pipeline(
            GradientBoostingRegressor(
                n_estimators=boosting_count,
                learning_rate=0.035,
                max_depth=2,
                loss="huber",
                random_state=random_state,
            )
        ),
        "HistGradientBoosting": pipeline(
            HistGradientBoostingRegressor(
                max_iter=boosting_count,
                learning_rate=0.04,
                l2_regularization=1.0,
                random_state=random_state,
            )
        ),
        "ExtraTrees_PowerTarget": TransformedTargetRegressor(
            regressor=pipeline(
                ExtraTreesRegressor(
                    n_estimators=tree_count,
                    max_features=1.0,
                    min_samples_leaf=1,
                    random_state=random_state + 1,
                    n_jobs=-1,
                )
            ),
            transformer=PowerTransformer(method="yeo-johnson", standardize=True),
        ),
        "SVR_RBF": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=10.0, epsilon=0.10, gamma="scale")),
            ]
        ),
    }

    if not fast:
        try:
            from xgboost import XGBRegressor

            candidates["XGBoost"] = pipeline(
                XGBRegressor(
                    n_estimators=900,
                    learning_rate=0.025,
                    max_depth=3,
                    min_child_weight=2,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    reg_alpha=0.05,
                    reg_lambda=2.0,
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=-1,
                )
            )
        except ImportError:
            pass

        try:
            from lightgbm import LGBMRegressor

            candidates["LightGBM"] = pipeline(
                LGBMRegressor(
                    n_estimators=800,
                    learning_rate=0.025,
                    num_leaves=15,
                    min_child_samples=12,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    reg_lambda=2.0,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                )
            )
        except ImportError:
            pass

        try:
            from catboost import CatBoostRegressor

            candidates["CatBoost"] = pipeline(
                CatBoostRegressor(
                    iterations=800,
                    depth=6,
                    learning_rate=0.03,
                    loss_function="RMSE",
                    random_seed=random_state,
                    verbose=False,
                    thread_count=-1,
                    allow_writing_files=False,
                )
            )
        except ImportError:
            pass

    return candidates


def make_cv_splitter(
    split_strategy: str,
    identifiers: pd.DataFrame,
    train_indices: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> tuple[object, np.ndarray | None]:
    """Build cross-validation folds matching the external split strategy."""
    if split_strategy == "grouped":
        smiles_column = next((column for column in ("SMILES", "smiles") if column in identifiers.columns), None)
        if smiles_column is None:
            raise ValueError("Grouped cross-validation requires a SMILES column.")
        groups = identifiers.iloc[train_indices][smiles_column].fillna("__missing_smiles__").astype(str).to_numpy()
        folds = min(cv_folds, len(np.unique(groups)))
        if folds < 2:
            raise ValueError("At least two distinct training SMILES groups are required.")
        return GroupKFold(n_splits=folds), groups

    folds = min(cv_folds, len(train_indices))
    if folds < 2:
        raise ValueError("At least two training rows are required for cross-validation.")
    return KFold(n_splits=folds, shuffle=True, random_state=random_state), None


def calibration_is_beneficial(
    y_true: np.ndarray,
    raw_predictions: np.ndarray,
    calibrated_predictions: np.ndarray,
) -> bool:
    """Require material R² gain without sacrificing QSAR agreement diagnostics.

    Calibration is estimated only from training out-of-fold predictions. A tiny R² gain is
    not enough to justify it because an affine transformation can worsen concordance and the
    origin-forced r_m² diagnostics even when squared error changes slightly.
    """
    raw_metrics = calculate_external_validation_metrics(y_true, raw_predictions, y_train=y_true)
    calibrated_metrics = calculate_external_validation_metrics(y_true, calibrated_predictions, y_train=y_true)
    if float(calibrated_metrics["R2_ext"]) < float(raw_metrics["R2_ext"]) + MIN_CALIBRATION_R2_GAIN:
        return False
    for metric in ("CCC_ext", "r_m^2", "Average r_m^2"):
        if float(calibrated_metrics[metric]) < float(raw_metrics[metric]) - MAX_CALIBRATION_AGREEMENT_LOSS:
            return False
    if (
        float(calibrated_metrics["Delta r_m^2"])
        > float(raw_metrics["Delta r_m^2"]) + MAX_CALIBRATION_DELTA_RM2_INCREASE
    ):
        return False
    return True


def fit_cross_validated_ensemble(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    cv: object,
    groups: np.ndarray | None,
    random_state: int,
    fast: bool = False,
) -> tuple[str, dict[str, object], object | None, dict[str, float]]:
    """Select and calibrate a model or OOF ensemble using training data only."""
    candidates = build_candidate_models(random_state=random_state, fast=fast)
    oof_predictions: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    accepted_calibrations: dict[str, bool] = {}

    for name, model in candidates.items():
        predictions = cross_val_predict(
            clone(model),
            x_train,
            y_train,
            cv=cv,
            groups=groups,
            method="predict",
            n_jobs=1,
        )
        oof_predictions[name] = np.asarray(predictions, dtype=float)
        scores[name] = float(r2_score(y_train, predictions))

        calibrator = LinearRegression()
        calibrator.fit(oof_predictions[name].reshape(-1, 1), y_train)
        calibrated_predictions = calibrator.predict(oof_predictions[name].reshape(-1, 1))
        calibrated_name = f"{name}_Calibrated"
        oof_predictions[calibrated_name] = np.asarray(calibrated_predictions, dtype=float)
        scores[calibrated_name] = float(r2_score(y_train, calibrated_predictions))
        accepted_calibrations[name] = calibration_is_beneficial(
            y_train,
            oof_predictions[name],
            oof_predictions[calibrated_name],
        )

    base_names = list(candidates)
    ordered_base_names = sorted(base_names, key=lambda name: scores[name], reverse=True)
    stack_names = ordered_base_names[: min(4, len(ordered_base_names))]
    stack_features = np.column_stack([oof_predictions[name] for name in stack_names])
    meta_model = RidgeCV(alphas=np.logspace(-6, 3, 25), fit_intercept=True)
    meta_model.fit(stack_features, y_train)
    ensemble_oof = np.asarray(meta_model.predict(stack_features), dtype=float)
    scores["OOF_Ridge_Ensemble"] = float(r2_score(y_train, ensemble_oof))

    ensemble_calibrator = LinearRegression()
    ensemble_calibrator.fit(ensemble_oof.reshape(-1, 1), y_train)
    calibrated_ensemble_oof = ensemble_calibrator.predict(ensemble_oof.reshape(-1, 1))
    scores["OOF_Ridge_Ensemble_Calibrated"] = float(r2_score(y_train, calibrated_ensemble_oof))
    ensemble_calibration_accepted = calibration_is_beneficial(
        y_train,
        ensemble_oof,
        calibrated_ensemble_oof,
    )

    preferred_candidates = [
        f"{name}_Calibrated" if accepted_calibrations[name] else name
        for name in base_names
    ]
    best_candidate_name = max(preferred_candidates, key=scores.get)
    best_ensemble_name = (
        "OOF_Ridge_Ensemble_Calibrated"
        if ensemble_calibration_accepted
        else "OOF_Ridge_Ensemble"
    )
    best_single_model_reference_score = max(
        max(scores[name], scores[f"{name}_Calibrated"])
        for name in base_names
    )
    if scores[best_ensemble_name] > best_single_model_reference_score + 0.005:
        selected_models = {name: clone(candidates[name]).fit(x_train, y_train) for name in stack_names}
        if best_ensemble_name.endswith("_Calibrated"):
            ensemble_bundle = {
                "stacker": meta_model,
                "calibrator": ensemble_calibrator,
            }
            return best_ensemble_name, selected_models, ensemble_bundle, scores
        return best_ensemble_name, selected_models, meta_model, scores

    if best_candidate_name.endswith("_Calibrated"):
        base_name = best_candidate_name.removesuffix("_Calibrated")
        selected_model = clone(candidates[base_name]).fit(x_train, y_train)
        calibrator = LinearRegression()
        calibrator.fit(oof_predictions[base_name].reshape(-1, 1), y_train)
        return best_candidate_name, {base_name: selected_model}, calibrator, scores

    selected_model = clone(candidates[best_candidate_name]).fit(x_train, y_train)
    return best_candidate_name, {best_candidate_name: selected_model}, None, scores


def build_validation_manifest(
    frame: pd.DataFrame,
    target_col: str,
    identifiers: pd.DataFrame,
    train_indices: np.ndarray,
    external_indices: np.ndarray,
    protocol: str,
    split_strategy: str,
    test_size: float,
    random_state: int,
    cv_folds: int,
    selected_model: str,
    cv_scores: dict[str, float],
    summary_table: pd.DataFrame,
) -> dict[str, object]:
    """Build a machine-readable audit record for the validation run."""
    smiles_column = next((column for column in ("SMILES", "smiles") if column in identifiers.columns), None)
    train_smiles: set[str] = set()
    external_smiles: set[str] = set()
    if smiles_column is not None:
        train_smiles = set(identifiers.iloc[train_indices][smiles_column].fillna("__missing_smiles__").astype(str))
        external_smiles = set(
            identifiers.iloc[external_indices][smiles_column].fillna("__missing_smiles__").astype(str)
        )
    overlap = train_smiles & external_smiles
    overlapping_external_rows = 0
    if smiles_column is not None:
        overlapping_external_rows = int(
            identifiers.iloc[external_indices][smiles_column].fillna("__missing_smiles__").astype(str).isin(overlap).sum()
        )
    repeated_smiles_rows = 0
    max_rows_per_smiles = 0
    if smiles_column is not None:
        counts = identifiers[smiles_column].fillna("__missing_smiles__").astype(str).value_counts()
        repeated_smiles_rows = int(counts[counts > 1].sum())
        max_rows_per_smiles = int(counts.max())

    thresholded = summary_table[summary_table["Result"].isin(["Pass", "Fail"])]
    return {
        "protocol": protocol,
        "protocol_interpretation": (
            "Strict molecule-group holdout with no SMILES overlap"
            if split_strategy == "grouped"
            else "Paper-reproduction random row holdout; not strict unseen-molecule external validation"
        ),
        "dataset": {
            "rows": int(len(frame)),
            "target_column": target_col,
            "unique_smiles": int(len(train_smiles | external_smiles)) if smiles_column else None,
            "rows_from_repeated_smiles": repeated_smiles_rows if smiles_column else None,
            "maximum_rows_for_one_smiles": max_rows_per_smiles if smiles_column else None,
        },
        "split": {
            "strategy": split_strategy,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "training_rows": int(len(train_indices)),
            "validation_rows": int(len(external_indices)),
            "training_unique_smiles": int(len(train_smiles)) if smiles_column else None,
            "validation_unique_smiles": int(len(external_smiles)) if smiles_column else None,
            "overlapping_smiles": int(len(overlap)) if smiles_column else None,
            "overlapping_validation_rows": overlapping_external_rows if smiles_column else None,
        },
        "model_selection": {
            "selected_model": selected_model,
            "cv_folds": int(cv_folds),
            "external_labels_used_for_selection": False,
            "calibration_policy": {
                "minimum_oof_r2_gain": MIN_CALIBRATION_R2_GAIN,
                "maximum_agreement_metric_loss": MAX_CALIBRATION_AGREEMENT_LOSS,
                "maximum_delta_rm2_increase": MAX_CALIBRATION_DELTA_RM2_INCREASE,
            },
            "cross_validated_r2": {name: float(score) for name, score in cv_scores.items()},
        },
        "acceptance_summary": {
            "criteria_passed": int((thresholded["Result"] == "Pass").sum()),
            "criteria_failed": int((thresholded["Result"] == "Fail").sum()),
            "all_thresholded_criteria_pass": bool((thresholded["Result"] == "Pass").all()),
        },
        "software": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def predict_selected_model(
    selected_name: str,
    fitted_models: dict[str, object],
    meta_model: object | None,
    x_external: pd.DataFrame,
) -> np.ndarray:
    """Predict external responses from the selected single model or stacked ensemble."""
    if selected_name in {"OOF_Ridge_Ensemble", "OOF_Ridge_Ensemble_Calibrated"}:
        if meta_model is None:
            raise ValueError("The stacked ensemble requires a fitted meta-model.")
        base_predictions = np.column_stack(
            [fitted_models[name].predict(x_external) for name in fitted_models]
        )
        if selected_name.endswith("_Calibrated"):
            stacker = meta_model["stacker"]
            calibrator = meta_model["calibrator"]
            stacked_predictions = np.asarray(stacker.predict(base_predictions), dtype=float)
            return np.asarray(calibrator.predict(stacked_predictions.reshape(-1, 1)), dtype=float)
        return np.asarray(meta_model.predict(base_predictions), dtype=float)
    if selected_name.endswith("_Calibrated"):
        if meta_model is None:
            raise ValueError("A calibrated model requires a fitted calibration model.")
        base_name = selected_name.removesuffix("_Calibrated")
        raw_predictions = np.asarray(fitted_models[base_name].predict(x_external), dtype=float)
        return np.asarray(meta_model.predict(raw_predictions.reshape(-1, 1)), dtype=float)
    return np.asarray(fitted_models[selected_name].predict(x_external), dtype=float)


def train_and_validate_external_model(
    dataset_csv_path: str | Path = "data/final/clean_trial4.csv",
    target_col: str | None = "logkpl",
    protocol: str = "strict-grouped",
    split_strategy: str | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    cv_folds: int = 5,
    output_dir: str | Path = ".",
    fast: bool = False,
) -> ValidationTrainingResult:
    """Train with holdout discipline, export validation CSVs, and run external metrics."""
    split_strategy, test_size, random_state = resolve_validation_protocol(
        protocol,
        split_strategy,
        test_size,
        random_state,
    )
    frame, resolved_target = load_modeling_dataset(dataset_csv_path, target_col=target_col)
    features, y, identifiers = prepare_modeling_data(frame, resolved_target)
    train_indices, external_indices = split_external_holdout(
        features,
        y,
        identifiers,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
    )
    cv, groups = make_cv_splitter(
        split_strategy,
        identifiers,
        train_indices,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    selected_name, fitted_models, meta_model, cv_scores = fit_cross_validated_ensemble(
        features.iloc[train_indices],
        y[train_indices],
        cv=cv,
        groups=groups,
        random_state=random_state,
        fast=fast,
    )
    prediction_values = predict_selected_model(
        selected_name,
        fitted_models,
        meta_model,
        features.iloc[external_indices],
    )

    train_data = pd.DataFrame({"Experimental_logKp": y[train_indices]})
    training_membership = identifiers.iloc[train_indices].reset_index(drop=True).copy()
    training_membership["Experimental_logKp"] = y[train_indices]
    external_predictions = identifiers.iloc[external_indices].reset_index(drop=True).copy()
    external_predictions["Experimental_logKp"] = y[external_indices]
    external_predictions["Predicted_logKp"] = prediction_values

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_data.csv"
    external_path = output_dir / "external_test_predictions.csv"
    train_data.to_csv(train_path, index=False)
    training_membership.to_csv(output_dir / "training_set_membership.csv", index=False)
    external_predictions.to_csv(external_path, index=False)
    joblib.dump(
        {
            "selected_model": selected_name,
            "models": fitted_models,
            "meta_model": meta_model,
            "feature_columns": list(features.columns),
            "protocol": protocol,
            "split_strategy": split_strategy,
            "test_size": test_size,
            "random_state": random_state,
            "cv_folds": cv_folds,
            "target_column": resolved_target,
            "scikit_learn_version": sklearn.__version__,
            "cv_scores": cv_scores,
        },
        output_dir / "external_validation_model.joblib",
    )

    print(f"Validation protocol: {protocol}")
    print(f"Split settings: {split_strategy}, test_size={test_size}, random_state={random_state}")
    if protocol == "paper-reproduction":
        print("Note: paper-reproduction uses a random row split and is not strict unseen-SMILES validation.")
    print(f"Selected model: {selected_name}")
    for name, score in sorted(cv_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"Training-only CV R2 - {name}: {score:.4f}")
    print("External labels were not used for model selection or calibration.")

    summary = run_external_validation(
        csv_path=external_path,
        actual_col="Experimental_logKp",
        pred_col="Predicted_logKp",
        train_csv_path=train_path,
        train_target_col="Experimental_logKp",
        output_dir=output_dir / "outputs" / "external_validation",
    )
    score_table = pd.DataFrame(
        sorted(cv_scores.items(), key=lambda item: item[1], reverse=True),
        columns=["Model", "Training-only OOF R2"],
    )
    score_table["Selected"] = score_table["Model"].eq(selected_name)
    score_table.to_csv(output_dir / "outputs" / "external_validation" / "model_selection_scores.csv", index=False)
    manifest = build_validation_manifest(
        frame=frame,
        target_col=resolved_target,
        identifiers=identifiers,
        train_indices=train_indices,
        external_indices=external_indices,
        protocol=protocol,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
        selected_model=selected_name,
        cv_scores=cv_scores,
        summary_table=summary,
    )
    manifest_path = output_dir / "outputs" / "external_validation" / "validation_protocol.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if split_strategy == "random" and manifest["split"]["overlapping_validation_rows"]:
        print(
            "Warning: paper-reproduction validation contains "
            f"{manifest['split']['overlapping_validation_rows']} validation rows whose SMILES also occur in training."
        )
    return ValidationTrainingResult(
        selected_model=selected_name,
        cross_validated_scores=cv_scores,
        train_data=train_data,
        external_predictions=external_predictions,
        summary_table=summary,
    )
