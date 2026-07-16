from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import warnings
from functools import partial

import joblib
import numpy as np
import pandas as pd

from ..config import load_yaml
from ..data_loading import load_bundle
from ..dependencies import MissingDependencyError, require_module
from ..metrics import regression_metrics
from ..paper_reconstruction import PAPER_METRICS
from ..paths import ProjectPaths
from ..splitters import naive_random_split
from .common import FittedRun, evaluate_regressor, save_predictions_plot, save_run_summary
from .transforms import CorrelationFilter

LOGGER = logging.getLogger(__name__)

sk_ensemble = require_module("sklearn.ensemble", "Install scikit-learn to run the benchmark pipeline.")
sk_feature_selection = require_module("sklearn.feature_selection", "Install scikit-learn to run the benchmark pipeline.")
sk_linear = require_module("sklearn.linear_model", "Install scikit-learn to run the benchmark pipeline.")
sk_model_selection = require_module("sklearn.model_selection", "Install scikit-learn to run the benchmark pipeline.")
sk_pipeline = require_module("sklearn.pipeline", "Install scikit-learn to run the benchmark pipeline.")
sk_preprocessing = require_module("sklearn.preprocessing", "Install scikit-learn to run the benchmark pipeline.")
sk_compose = require_module("sklearn.compose", "Install scikit-learn to run the benchmark pipeline.")
sk_svm = require_module("sklearn.svm", "Install scikit-learn to run the benchmark pipeline.")
sk_tags = require_module("sklearn.utils._tags", "Install scikit-learn to run the benchmark pipeline.")
sk_exceptions = require_module("sklearn.exceptions", "Install scikit-learn to run the benchmark pipeline.")

Pipeline = sk_pipeline.Pipeline
RandomizedSearchCV = sk_model_selection.RandomizedSearchCV
RepeatedKFold = sk_model_selection.RepeatedKFold
SelectKBest = sk_feature_selection.SelectKBest
mutual_info_regression = sk_feature_selection.mutual_info_regression
PowerTransformer = sk_preprocessing.PowerTransformer
RobustScaler = sk_preprocessing.RobustScaler
TransformedTargetRegressor = sk_compose.TransformedTargetRegressor
ExtraTreesRegressor = sk_ensemble.ExtraTreesRegressor
HistGradientBoostingRegressor = sk_ensemble.HistGradientBoostingRegressor
RandomForestRegressor = sk_ensemble.RandomForestRegressor
GradientBoostingRegressor = sk_ensemble.GradientBoostingRegressor
StackingRegressor = sk_ensemble.StackingRegressor
ElasticNet = sk_linear.ElasticNet
RidgeCV = sk_linear.RidgeCV
SVR = sk_svm.SVR
ConvergenceWarning = sk_exceptions.ConvergenceWarning


def _patch_xgboost_sklearn_tags(estimator_cls: Any) -> Any:
    """Make older XGBoost releases compatible with scikit-learn >= 1.6."""
    def xgboost_tags(self):
        return sk_tags.default_tags(self)

    needs_patch = False
    try:
        needs_patch = sk_tags.get_tags(estimator_cls()).estimator_type != "regressor"
    except AttributeError:
        needs_patch = True
    if needs_patch:
        LOGGER.warning(
            "Patching %s for scikit-learn tag compatibility. Upgrade xgboost to >=2.1.4 to avoid this shim.",
            estimator_cls.__name__,
        )
        for klass in estimator_cls.mro():
            if getattr(klass, "__module__", "").startswith("xgboost."):
                klass.__sklearn_tags__ = xgboost_tags
    return estimator_cls


def _optional_estimator(module_name: str, class_name: str, install_hint: str):
    module = require_module(module_name, install_hint)
    estimator_cls = getattr(module, class_name)
    if module_name == "xgboost" and class_name == "XGBRegressor":
        estimator_cls = _patch_xgboost_sklearn_tags(estimator_cls)
    return estimator_cls


def benchmark_estimators(random_state: int, estimator_n_jobs: int = 1) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    estimators: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "ExtraTrees": (
            ExtraTreesRegressor(random_state=random_state, n_estimators=800, n_jobs=estimator_n_jobs),
            {
                "model__n_estimators": [400, 800, 1200],
                "model__max_depth": [None, 12, 20, 30],
                "model__max_features": ["sqrt", 0.5, 0.8, 1.0],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(random_state=random_state),
            {
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_depth": [None, 3, 5, 8],
                "model__max_leaf_nodes": [15, 31, 63, 127],
                "model__min_samples_leaf": [5, 10, 20],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=random_state),
            {
                "model__n_estimators": [200, 400, 800],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4, 5],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        ),
        "RF": (
            RandomForestRegressor(random_state=random_state, n_jobs=estimator_n_jobs),
            {
                "model__n_estimators": [400, 800, 1200],
                "model__max_depth": [None, 10, 16, 24],
                "model__max_features": ["sqrt", 0.5, 0.8, 1.0],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "ElasticNet": (
            ElasticNet(random_state=random_state, max_iter=20000),
            {
                "model__alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        ),
        "SVR (RBF)": (
            SVR(kernel="rbf"),
            {
                "model__C": [3, 5, 10, 20, 40],
                "model__gamma": ["scale", 0.005, 0.01, 0.03, 0.05],
                "model__epsilon": [0.03, 0.05, 0.1, 0.2],
            },
        ),
    }

    optional_specs = {
        "XGBoost": (
            "xgboost",
            "XGBRegressor",
            "Install xgboost to run the benchmark pipeline.",
            dict(random_state=random_state, objective="reg:squarederror", tree_method="hist", verbosity=0, n_jobs=estimator_n_jobs),
            {
                "model__n_estimators": [400, 800, 1200],
                "model__learning_rate": [0.01, 0.03, 0.05],
                "model__max_depth": [3, 4, 6, 8],
                "model__min_child_weight": [1, 3, 5],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.5, 0.7, 0.9, 1.0],
                "model__reg_lambda": [1.0, 5.0, 10.0],
            },
        ),
        "CatBoost": (
            "catboost",
            "CatBoostRegressor",
            "Install catboost to run the benchmark pipeline.",
            dict(random_state=random_state, verbose=0, allow_writing_files=False, thread_count=estimator_n_jobs),
            {
                "model__iterations": [400, 800, 1200],
                "model__learning_rate": [0.01, 0.03, 0.05],
                "model__depth": [4, 6, 8, 10],
                "model__l2_leaf_reg": [1, 3, 5, 7, 9],
                "model__bagging_temperature": [0.0, 0.5, 1.0],
            },
        ),
        "LGBM": (
            "lightgbm",
            "LGBMRegressor",
            "Install lightgbm to run the benchmark pipeline.",
            dict(random_state=random_state, verbose=-1, n_jobs=estimator_n_jobs),
            {
                "model__n_estimators": [400, 800, 1200],
                "model__learning_rate": [0.01, 0.03, 0.05],
                "model__num_leaves": [15, 31, 63, 127],
                "model__min_child_samples": [5, 10, 20],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.5, 0.7, 0.9, 1.0],
                "model__reg_lambda": [0.0, 0.1, 1.0, 5.0],
            },
        ),
    }
    for model_name, (module_name, class_name, hint, kwargs, grid) in optional_specs.items():
        try:
            estimator_cls = _optional_estimator(module_name, class_name, hint)
        except MissingDependencyError:
            LOGGER.warning("Skipping optional benchmark model %s because its dependency is unavailable.", model_name)
            continue
        estimators[model_name] = (estimator_cls(**kwargs), grid)
    return estimators


def build_benchmark_estimator(
    model_name: str,
    estimator: Any,
    correlation_threshold: float,
    correlation_threshold_options: list[float] | None,
    selector_k: list[Any],
    use_target_transform: bool,
    random_state: int,
) -> tuple[Any, dict[str, list[Any]]]:
    steps: list[tuple[str, Any]] = [("corr", CorrelationFilter(threshold=correlation_threshold))]
    if model_name in {"ElasticNet", "SVR (RBF)"}:
        steps.append(("scaler", RobustScaler()))
    mi_score = partial(mutual_info_regression, random_state=random_state)
    steps.extend(
        [
            ("selector", SelectKBest(score_func=mi_score, k="all")),
            ("model", estimator),
        ]
    )
    pipeline = Pipeline(steps)
    param_prefix = ""
    wrapped_estimator: Any = pipeline
    if use_target_transform:
        wrapped_estimator = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=PowerTransformer(method="yeo-johnson", standardize=True),
        )
        param_prefix = "regressor__"
    corr_options = correlation_threshold_options or [0.95, 0.98, correlation_threshold]
    params = {
        f"{param_prefix}corr__threshold": corr_options,
        f"{param_prefix}selector__k": selector_k,
    }
    return wrapped_estimator, params


def _as_search_values(values: Any) -> list[Any]:
    if isinstance(values, list):
        return values
    return [values]


def _cv_metric(search: RandomizedSearchCV) -> float:
    return float(abs(search.best_score_))


def _search(
    estimator: Any,
    params: dict[str, list[Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_iter: int,
    scoring: str,
    cv,
    n_jobs: int,
) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=min(n_iter, max(1, int(np.prod([len(values) for values in params.values()])))),
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search


def _compare_with_paper(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, reference in PAPER_METRICS.items():
        if model_name not in metrics_frame["model"].values:
            continue
        observed = metrics_frame.loc[metrics_frame["model"] == model_name].iloc[0]
        row = {"model": model_name}
        for metric_name, reference_value in reference.items():
            if metric_name not in observed:
                continue
            row[f"{metric_name}_paper"] = reference_value
            row[f"{metric_name}_observed"] = float(observed[metric_name])
            row[f"{metric_name}_delta"] = float(observed[metric_name]) - float(reference_value)
        rows.append(row)
    return pd.DataFrame(rows)


def run_benchmark(paths: ProjectPaths, config_path: Path) -> pd.DataFrame:
    config = load_yaml(config_path)
    output_dir = paths.models / "reproduction" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(paths)
    data = bundle.clean_trial4.copy()
    feature_columns = [column for column in data.columns if column not in {"logkpl", "Compound", "SMILES"}]
    x = data[feature_columns]
    y = data["logkpl"]

    split = naive_random_split(data, test_size=config["test_size"], random_state=config["random_state"])
    x_train = x.iloc[split.train_index]
    x_test = x.iloc[split.test_index]
    y_train = y.iloc[split.train_index]
    y_test = y.iloc[split.test_index]

    cv = RepeatedKFold(
        n_splits=config["cv_folds"],
        n_repeats=config["cv_repeats"],
        random_state=config["random_state"],
    )

    selector_k = config.get("selector_k_options", [40, 60, 80, 100, 120, "all"])
    search_n_jobs = int(config.get("n_jobs", 1))
    estimator_n_jobs = int(config.get("estimator_n_jobs", 1))
    disabled_models = set(config.get("disabled_models", []))
    model_param_overrides = config.get("model_param_overrides", {})
    model_selector_k_options = config.get("model_selector_k_options", {})
    correlation_threshold_options = config.get("correlation_threshold_options")
    skip_post_search_cv = bool(config.get("skip_post_search_cv", False))
    run_ensembles = bool(config.get("run_ensembles", True))
    runs: list[FittedRun] = []
    searches: dict[str, RandomizedSearchCV] = {}

    for model_name, (raw_estimator, model_grid) in benchmark_estimators(config["random_state"], estimator_n_jobs).items():
        if model_name in disabled_models:
            LOGGER.info("Skipping disabled benchmark model %s.", model_name)
            continue
        LOGGER.info("Tuning benchmark model %s.", model_name)
        estimator, shared_grid = build_benchmark_estimator(
            model_name=model_name,
            estimator=raw_estimator,
            correlation_threshold=config["correlation_threshold"],
            correlation_threshold_options=correlation_threshold_options,
            selector_k=model_selector_k_options.get(model_name, selector_k),
            use_target_transform=config.get("use_target_transform", True),
            random_state=config.get("feature_selection_random_state", config["random_state"]),
        )
        model_grid = model_param_overrides.get(model_name, model_grid)
        prefixed_grid = {}
        prefix = "regressor__" if config.get("use_target_transform", True) else ""
        for key, values in model_grid.items():
            prefixed_grid[f"{prefix}{key}"] = _as_search_values(values)
        with warnings.catch_warnings():
            if model_name == "ElasticNet":
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            search = _search(
                estimator=estimator,
                params={**shared_grid, **prefixed_grid},
                x_train=x_train,
                y_train=y_train,
                random_state=config["random_state"],
                n_iter=config["search_iterations"],
                scoring=config["optimization_scoring"],
                cv=cv,
                n_jobs=search_n_jobs,
            )
            searches[model_name] = search
            if skip_post_search_cv:
                predictions = search.best_estimator_.predict(x_test)
                metrics = regression_metrics(y_test, predictions).__dict__
                metrics["cv_rmse"] = _cv_metric(search)
                metrics["mae_ci_low"] = float("nan")
                metrics["mae_ci_high"] = float("nan")
                metrics["cv_mae"] = float("nan")
            else:
                predictions, metrics = evaluate_regressor(
                    search.best_estimator_,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    cv_folds=config["cv_folds"],
                    random_state=config["random_state"],
                    cv=cv,
                    cv_scoring="neg_root_mean_squared_error",
                    cv_metric_name="cv_rmse",
                )
                metrics["cv_mae"] = float(
                    abs(
                        sk_model_selection.cross_val_score(
                            search.best_estimator_,
                            x_train,
                            y_train,
                            scoring="neg_mean_absolute_error",
                            cv=cv,
                            n_jobs=search_n_jobs,
                        ).mean()
                    )
                )
        metrics["tuning_rmse"] = _cv_metric(search)
        artifact_path = output_dir / f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(search.best_estimator_, artifact_path)
        save_predictions_plot(
            y_test,
            predictions,
            paths.figures / "benchmark" / f"{model_name.replace(' ', '_')}_pred_vs_actual.png",
            f"{model_name} Benchmark Predicted vs Actual",
        )
        runs.append(FittedRun(model_name, search.best_estimator_, predictions, metrics, artifact_path))
        LOGGER.info("Finished %s: test R2 %.4f, test RMSE %.4f.", model_name, metrics["r2"], metrics["rmse"])

    strong_names = [name for name in ["XGBoost", "CatBoost", "LGBM", "ExtraTrees", "HistGradientBoosting"] if name in searches]
    if run_ensembles and len(strong_names) >= 2:
        stack = StackingRegressor(
            estimators=[(name, searches[name].best_estimator_) for name in strong_names],
            final_estimator=RidgeCV(alphas=(0.1, 1.0, 10.0)),
            passthrough=True,
            n_jobs=search_n_jobs,
        )
        predictions, metrics = evaluate_regressor(
            stack,
            x_train,
            y_train,
            x_test,
            y_test,
            cv_folds=config["cv_folds"],
            random_state=config["random_state"],
            cv=cv,
            cv_scoring="neg_root_mean_squared_error",
            cv_metric_name="cv_rmse",
        )
        metrics["cv_mae"] = float(
            abs(
                sk_model_selection.cross_val_score(
                    stack,
                    x_train,
                    y_train,
                    scoring="neg_mean_absolute_error",
                    cv=cv,
                    n_jobs=search_n_jobs,
                ).mean()
            )
        )
        metrics["tuning_rmse"] = metrics["cv_rmse"]
        artifact_path = output_dir / "StackingRegressor.joblib"
        joblib.dump(stack, artifact_path)
        save_predictions_plot(
            y_test,
            predictions,
            paths.figures / "benchmark" / "StackingRegressor_pred_vs_actual.png",
            "Benchmark Stacking Regressor",
        )
        runs.append(FittedRun("Stacking Regressor", stack, predictions, metrics, artifact_path))

        mean_predictions = np.mean([searches[name].best_estimator_.predict(x_test) for name in strong_names], axis=0)
        ensemble_metrics = regression_metrics(y_test, mean_predictions).__dict__
        ensemble_metrics["cv_rmse"] = float("nan")
        ensemble_metrics["cv_mae"] = float("nan")
        ensemble_metrics["tuning_rmse"] = float(
            np.mean([_cv_metric(searches[name]) for name in strong_names])
        )
        ensemble_path = output_dir / "MeanEnsemble.json"
        ensemble_path.write_text(json.dumps({"models": strong_names}, indent=2), encoding="utf-8")
        save_predictions_plot(
            y_test,
            mean_predictions,
            paths.figures / "benchmark" / "MeanEnsemble_pred_vs_actual.png",
            "Benchmark Mean Ensemble",
        )
        runs.append(FittedRun("Mean Ensemble", strong_names, mean_predictions, ensemble_metrics, ensemble_path))

    metrics_frame = pd.DataFrame(
        [{"model": run.model_name, "artifact_path": str(run.artifact_path), **run.metrics} for run in runs]
    ).sort_values(["rmse", "r2", "mae"], ascending=[True, False, True])
    metrics_frame.to_csv(paths.reports / "tables" / "benchmark_metrics.csv", index=False)

    comparison = _compare_with_paper(metrics_frame)
    if not comparison.empty:
        comparison.to_csv(paths.reports / "tables" / "benchmark_vs_paper_comparison.csv", index=False)

    fitted_runs = [run for run in runs if hasattr(run.estimator, "predict")]
    best_cv_run = min(fitted_runs, key=lambda run: (run.metrics.get("tuning_rmse", float("inf")), run.metrics["rmse"]))
    best_model_target = output_dir / "best_benchmark_model.joblib"
    joblib.dump(best_cv_run.estimator, best_model_target)
    (output_dir / "best_benchmark_model.json").write_text(
        json.dumps(
            {
                "model": best_cv_run.model_name,
                "selection_basis": "lowest repeated-CV RMSE on the training set",
                "metrics": best_cv_run.metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics_frame
