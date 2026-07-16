from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import load_yaml
from ..data_loading import load_bundle
from ..dependencies import MissingDependencyError, require_module
from ..metrics import regression_metrics
from ..paths import ProjectPaths
from ..paper_reconstruction import PAPER_METRICS
from ..splitters import naive_random_split
from .ann import build_baseline_ann, save_ann, train_ann
from .common import FittedRun, StandardScaler, evaluate_regressor, save_predictions_plot, save_run_summary

LOGGER = logging.getLogger(__name__)

sk_linear = require_module("sklearn.linear_model", "Install scikit-learn to train baseline models.")
sk_tree = require_module("sklearn.tree", "Install scikit-learn to train baseline models.")
sk_ensemble = require_module("sklearn.ensemble", "Install scikit-learn to train baseline models.")
sk_svm = require_module("sklearn.svm", "Install scikit-learn to train baseline models.")
sk_feature_selection = require_module("sklearn.feature_selection", "Install scikit-learn to train baseline models.")

LinearRegression = sk_linear.LinearRegression
Lasso = sk_linear.Lasso
DecisionTreeRegressor = sk_tree.DecisionTreeRegressor
RandomForestRegressor = sk_ensemble.RandomForestRegressor
GradientBoostingRegressor = sk_ensemble.GradientBoostingRegressor
SVR = sk_svm.SVR
SequentialFeatureSelector = sk_feature_selection.SequentialFeatureSelector


def _optional_estimator(module_name: str, class_name: str, install_hint: str):
    module = require_module(module_name, install_hint)
    return getattr(module, class_name)


def baseline_estimators(random_state: int) -> dict[str, object]:
    estimators = {
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "RF": RandomForestRegressor(random_state=random_state, n_estimators=300),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "SVR (RBF)": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        "Lasso": Lasso(alpha=0.001, random_state=random_state, max_iter=5000),
    }
    estimators["XGBoost"] = _optional_estimator("xgboost", "XGBRegressor", "Install xgboost to run the paper baseline.")(
        random_state=random_state,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="reg:squarederror",
    )
    estimators["CatBoost"] = _optional_estimator("catboost", "CatBoostRegressor", "Install catboost to run the paper baseline.")(
        random_state=random_state,
        verbose=0,
        allow_writing_files=False,
    )
    estimators["LGBM"] = _optional_estimator("lightgbm", "LGBMRegressor", "Install lightgbm to run the paper baseline.")(
        random_state=random_state
    )
    return estimators


def prepare_paper_baseline_data(bundle, config: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data = bundle.clean_trial4.copy()
    feature_columns = [column for column in data.columns if column not in {"logkpl", "Compound", "SMILES"}]
    x = data[feature_columns]
    y = data["logkpl"]
    groups = data["SMILES"]
    if config.get("scale_before_split", True):
        scaler = StandardScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=feature_columns, index=x.index)
        joblib.dump(scaler, config["scaler_output"])
    return x, y, groups


def fit_mlr_with_feature_selection(x_train, y_train, x_test, y_test, random_state: int) -> tuple[object, np.ndarray, dict[str, float], list[str]]:
    selector = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=10,
        direction="forward",
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
    )
    selector.fit(x_train, y_train)
    selected_columns = list(x_train.columns[selector.get_support()])
    estimator = LinearRegression().fit(x_train[selected_columns], y_train)
    predictions = estimator.predict(x_test[selected_columns])
    metrics = regression_metrics(y_test, predictions).__dict__
    cv_scores = require_module("sklearn.model_selection").cross_val_score(
        estimator,
        x_train[selected_columns],
        y_train,
        scoring="neg_mean_absolute_error",
        cv=5,
    )
    metrics["cv_mae"] = float(np.abs(cv_scores.mean()))
    return estimator, predictions, metrics, selected_columns


def compare_with_paper(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    paper_rows = []
    for model_name, expected in PAPER_METRICS.items():
        if model_name not in metrics_frame["model"].values:
            continue
        observed = metrics_frame.loc[metrics_frame["model"] == model_name].iloc[0]
        row = {"model": model_name}
        for metric_name, expected_value in expected.items():
            observed_value = float(observed[metric_name])
            row[f"{metric_name}_paper"] = expected_value
            row[f"{metric_name}_observed"] = observed_value
            row[f"{metric_name}_delta"] = observed_value - expected_value
        paper_rows.append(row)
    return pd.DataFrame(paper_rows)


def run_baseline(paths: ProjectPaths, config_path: Path) -> pd.DataFrame:
    config = load_yaml(config_path)
    output_dir = paths.models / "reproduction" / "paper_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    config["scaler_output"] = str(output_dir / "paper_baseline_scaler.joblib")
    bundle = load_bundle(paths)
    x, y, _ = prepare_paper_baseline_data(bundle, config)
    split = naive_random_split(bundle.clean_trial4, test_size=config["test_size"], random_state=config["random_state"])
    x_train = x.iloc[split.train_index]
    x_test = x.iloc[split.test_index]
    y_train = y.iloc[split.train_index]
    y_test = y.iloc[split.test_index]

    runs: list[FittedRun] = []
    estimator, predictions, metrics, selected_columns = fit_mlr_with_feature_selection(
        x_train, y_train, x_test, y_test, random_state=config["random_state"]
    )
    mlr_path = output_dir / "MLR_10_features.joblib"
    joblib.dump({"estimator": estimator, "selected_columns": selected_columns}, mlr_path)
    save_predictions_plot(y_test, predictions, paths.figures / "paper_baseline" / "MLR_10_features_pred_vs_actual.png", "MLR (10 features)")
    runs.append(FittedRun("MLR (10 features)", estimator, predictions, metrics, mlr_path))

    for model_name, estimator in baseline_estimators(config["random_state"]).items():
        predictions, metrics = evaluate_regressor(
            estimator,
            x_train,
            y_train,
            x_test,
            y_test,
            cv_folds=config["cv_folds"],
            random_state=config["random_state"],
        )
        artifact_path = output_dir / f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(estimator, artifact_path)
        save_predictions_plot(
            y_test,
            predictions,
            paths.figures / "paper_baseline" / f"{model_name.replace(' ', '_')}_pred_vs_actual.png",
            f"{model_name} Predicted vs Actual",
        )
        runs.append(FittedRun(model_name, estimator, predictions, metrics, artifact_path))

    if config.get("run_ann", True):
        ann_dir = output_dir / "ANN"
        ann_dir.mkdir(parents=True, exist_ok=True)
        ann = build_baseline_ann(input_dim=x_train.shape[1], learning_rate=config["ann"]["learning_rate"], dropout_rate=config["ann"]["dropout"])
        train_ann(
            ann,
            x_train.to_numpy(),
            y_train.to_numpy(),
            x_test.to_numpy(),
            y_test.to_numpy(),
            epochs=config["ann"]["epochs"],
            batch_size=config["ann"]["batch_size"],
            patience=config["ann"]["patience"],
            random_state=config["random_state"],
            output_dir=ann_dir,
        )
        ann_predictions = ann.predict(x_test.to_numpy(), verbose=0).reshape(-1)
        ann_metrics = regression_metrics(y_test, ann_predictions).__dict__
        ann_metrics["cv_mae"] = float("nan")
        save_ann(ann, ann_dir / "ann_model")
        save_predictions_plot(y_test, ann_predictions, paths.figures / "paper_baseline" / "ANN_pred_vs_actual.png", "ANN Predicted vs Actual")
        runs.append(FittedRun("ANN", ann, ann_predictions, ann_metrics, ann_dir / "ann_model.keras"))

    metrics_frame = save_run_summary(runs, paths.reports / "tables" / "paper_baseline_metrics.csv")
    comparison = compare_with_paper(metrics_frame)
    comparison.to_csv(paths.reports / "tables" / "paper_vs_baseline_comparison.csv", index=False)
    return metrics_frame
