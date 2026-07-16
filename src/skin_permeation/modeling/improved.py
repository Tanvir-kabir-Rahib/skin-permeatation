from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import load_yaml
from ..data_loading import load_bundle
from ..dependencies import require_module
from ..metrics import regression_metrics
from ..paths import ProjectPaths
from ..splitters import grouped_split, make_groups, naive_random_split
from .common import FittedRun, StandardScaler, evaluate_regressor, save_predictions_plot, save_run_summary

LOGGER = logging.getLogger(__name__)

sk_compose = require_module("sklearn.compose", "Install scikit-learn to run the improved pipeline.")
sk_pipeline = require_module("sklearn.pipeline", "Install scikit-learn to run the improved pipeline.")
sk_model_selection = require_module("sklearn.model_selection", "Install scikit-learn to run the improved pipeline.")
sk_feature_selection = require_module("sklearn.feature_selection", "Install scikit-learn to run the improved pipeline.")
sk_impute = require_module("sklearn.impute", "Install scikit-learn to run the improved pipeline.")
sk_linear = require_module("sklearn.linear_model", "Install scikit-learn to run the improved pipeline.")
sk_tree = require_module("sklearn.tree", "Install scikit-learn to run the improved pipeline.")
sk_ensemble = require_module("sklearn.ensemble", "Install scikit-learn to run the improved pipeline.")
sk_svm = require_module("sklearn.svm", "Install scikit-learn to run the improved pipeline.")

Pipeline = sk_pipeline.Pipeline
RandomizedSearchCV = sk_model_selection.RandomizedSearchCV
RepeatedKFold = sk_model_selection.RepeatedKFold
GroupKFold = sk_model_selection.GroupKFold
SimpleImputer = sk_impute.SimpleImputer
VarianceThreshold = sk_feature_selection.VarianceThreshold
LinearRegression = sk_linear.LinearRegression
Lasso = sk_linear.Lasso
DecisionTreeRegressor = sk_tree.DecisionTreeRegressor
RandomForestRegressor = sk_ensemble.RandomForestRegressor
GradientBoostingRegressor = sk_ensemble.GradientBoostingRegressor
SVR = sk_svm.SVR
StackingRegressor = sk_ensemble.StackingRegressor


def _optional_estimator(module_name: str, class_name: str, install_hint: str):
    module = require_module(module_name, install_hint)
    return getattr(module, class_name)


def improved_estimators(random_state: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    estimators: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Decision Tree": (
            DecisionTreeRegressor(random_state=random_state),
            {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 2, 4, 8]},
        ),
        "RF": (
            RandomForestRegressor(random_state=random_state),
            {"model__n_estimators": [200, 400, 800], "model__max_depth": [None, 8, 12], "model__min_samples_leaf": [1, 2, 4]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=random_state),
            {"model__n_estimators": [100, 200, 400], "model__learning_rate": [0.01, 0.05, 0.1], "model__max_depth": [2, 3, 4]},
        ),
        "SVR (RBF)": (
            SVR(kernel="rbf"),
            {"model__C": [1, 5, 10, 20], "model__gamma": ["scale", 0.01, 0.05], "model__epsilon": [0.05, 0.1, 0.2]},
        ),
        "Lasso": (
            Lasso(random_state=random_state, max_iter=5000),
            {"model__alpha": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]},
        ),
        "MLR": (LinearRegression(), {}),
    }
    estimators["XGBoost"] = (
        _optional_estimator("xgboost", "XGBRegressor", "Install xgboost to run the improved pipeline.")(
            random_state=random_state,
            objective="reg:squarederror",
        ),
        {"model__n_estimators": [200, 400, 800], "model__learning_rate": [0.01, 0.03, 0.1], "model__max_depth": [3, 5, 7]},
    )
    estimators["CatBoost"] = (
        _optional_estimator("catboost", "CatBoostRegressor", "Install catboost to run the improved pipeline.")(
            random_state=random_state,
            verbose=0,
            allow_writing_files=False,
        ),
        {"model__depth": [4, 6, 8], "model__learning_rate": [0.01, 0.03, 0.1], "model__iterations": [200, 400, 800]},
    )
    estimators["LGBM"] = (
        _optional_estimator("lightgbm", "LGBMRegressor", "Install lightgbm to run the improved pipeline.")(
            random_state=random_state
        ),
        {"model__n_estimators": [200, 400, 800], "model__learning_rate": [0.01, 0.03, 0.1], "model__num_leaves": [15, 31, 63]},
    )
    return estimators


def build_pipeline(model_name: str, estimator: Any) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
    ]
    if model_name in {"MLR", "Lasso", "SVR (RBF)"}:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def search_model(
    model_name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series | None,
    random_state: int,
    n_iter: int,
) -> Any:
    pipeline = build_pipeline(model_name, estimator)
    if not param_grid:
        return pipeline.fit(x_train, y_train)
    cv = GroupKFold(n_splits=5) if groups is not None else RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=min(n_iter, max(1, np.prod([len(values) for values in param_grid.values()]))),
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    fit_kwargs = {"groups": groups} if groups is not None else {}
    search.fit(x_train, y_train, **fit_kwargs)
    return search.best_estimator_


def descriptor_family_ablation(x: pd.DataFrame) -> dict[str, list[str]]:
    families = {
        "lipophilicity": [column for column in x.columns if "logp" in column.lower() or "xlogp" in column.lower()],
        "hydrogen_bonding": [column for column in x.columns if "hbd" in column.lower() or "hbacc" in column.lower()],
        "topology": [column for column in x.columns if column.startswith(("ATS", "BCUT", "WTPT", "WPATH", "Zagreb"))],
        "lipinski": [column for column in x.columns if "lipinski" in column.lower()],
        "surface_area": [column for column in x.columns if "psa" in column.lower() or "vabc" in column.lower()],
    }
    return {name: columns for name, columns in families.items() if columns}


def error_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["MW", "XLogP", "TopoPSA"]:
        if column not in frame.columns:
            continue
        bands = pd.qcut(frame[column], q=4, duplicates="drop")
        summary = frame.groupby(bands)["absolute_error"].agg(["count", "mean", "median"]).reset_index()
        summary["band_variable"] = column
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_improved(paths: ProjectPaths, config_path: Path) -> pd.DataFrame:
    config = load_yaml(config_path)
    output_dir = paths.models / "reproduction" / "improved"
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(paths)
    data = bundle.clean_trial4.copy()
    feature_columns = [column for column in data.columns if column not in {"logkpl", "Compound", "SMILES"}]
    x = data[feature_columns]
    y = data["logkpl"]
    groups = make_groups(data)

    split = grouped_split(data, groups, test_size=config["test_size"], random_state=config["random_state"])
    x_train = x.iloc[split.train_index]
    x_test = x.iloc[split.test_index]
    y_train = y.iloc[split.train_index]
    y_test = y.iloc[split.test_index]
    train_groups = groups.iloc[split.train_index]

    runs: list[FittedRun] = []
    for model_name, (estimator, param_grid) in improved_estimators(config["random_state"]).items():
        best_estimator = search_model(
            model_name,
            estimator,
            param_grid,
            x_train,
            y_train,
            groups=train_groups,
            random_state=config["random_state"],
            n_iter=config["search_iterations"],
        )
        predictions, metrics = evaluate_regressor(
            best_estimator,
            x_train,
            y_train,
            x_test,
            y_test,
            cv_folds=config["cv_folds"],
            random_state=config["random_state"],
        )
        artifact_path = output_dir / f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(best_estimator, artifact_path)
        save_predictions_plot(
            y_test,
            predictions,
            paths.figures / "improved" / f"{model_name.replace(' ', '_')}_pred_vs_actual.png",
            f"{model_name} Improved Predicted vs Actual",
        )
        runs.append(FittedRun(model_name, best_estimator, predictions, metrics, artifact_path))

    metrics_frame = save_run_summary(runs, paths.reports / "tables" / "improved_metrics.csv")
    top_models = metrics_frame.nsmallest(3, "mae")["model"].tolist()
    estimators_for_stack = [(run.model_name, run.estimator) for run in runs if run.model_name in top_models]
    if len(estimators_for_stack) >= 2:
        stack = StackingRegressor(estimators=estimators_for_stack, final_estimator=LinearRegression(), n_jobs=-1)
        predictions, metrics = evaluate_regressor(
            stack,
            x_train,
            y_train,
            x_test,
            y_test,
            cv_folds=config["cv_folds"],
            random_state=config["random_state"],
        )
        artifact_path = output_dir / "StackingRegressor.joblib"
        joblib.dump(stack, artifact_path)
        save_predictions_plot(y_test, predictions, paths.figures / "improved" / "StackingRegressor_pred_vs_actual.png", "Stacking Regressor")
        runs.append(FittedRun("Stacking Regressor", stack, predictions, metrics, artifact_path))
        metrics_frame = save_run_summary(runs, paths.reports / "tables" / "improved_metrics.csv")

    best_run = min(runs, key=lambda run: run.metrics["mae"])
    errors = x_test.copy()
    errors["actual_logkpl"] = y_test.to_numpy()
    errors["predicted_logkpl"] = best_run.predictions
    errors["absolute_error"] = np.abs(errors["actual_logkpl"] - errors["predicted_logkpl"])
    errors["SMILES"] = data.iloc[split.test_index]["SMILES"].to_numpy()
    errors["Compound"] = data.iloc[split.test_index]["Compound"].to_numpy()
    errors.to_csv(paths.reports / "tables" / "improved_test_errors.csv", index=False)
    ablation = descriptor_family_ablation(x_train)
    pd.DataFrame(
        [{"family": family, "descriptor_count": len(columns), "descriptors": ", ".join(columns)} for family, columns in ablation.items()]
    ).to_csv(paths.reports / "tables" / "descriptor_family_ablation_plan.csv", index=False)
    analysis_frame = error_analysis(errors)
    if not analysis_frame.empty:
        analysis_frame.to_csv(paths.reports / "tables" / "error_analysis_by_band.csv", index=False)
    return metrics_frame
