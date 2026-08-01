from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from ..artifacts import save_json, save_table
from ..dependencies import require_module
from ..metrics import bootstrap_metric_interval, mean_absolute_error, regression_metrics
from ..paths import ProjectPaths

sk_model_selection = require_module("sklearn.model_selection", "Install scikit-learn to train regression models.")
sk_preprocessing = require_module("sklearn.preprocessing", "Install scikit-learn to train regression models.")

KFold = sk_model_selection.KFold
cross_val_score = sk_model_selection.cross_val_score
StandardScaler = sk_preprocessing.StandardScaler

LOGGER = logging.getLogger(__name__)


_RUNTIME_CACHE = Path(tempfile.gettempdir()) / "skin_permeation_runtime_cache"
(_RUNTIME_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
(_RUNTIME_CACHE / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_RUNTIME_CACHE / "xdg"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


@dataclass
class FittedRun:
    model_name: str
    estimator: Any
    predictions: np.ndarray
    metrics: dict[str, float]
    artifact_path: Path


def save_estimator(estimator: Any, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(estimator, "save") and destination.suffix == ".keras":
        estimator.save(destination)
    else:
        joblib.dump(estimator, destination)
    return destination


def save_predictions_plot(y_true: pd.Series, y_pred: np.ndarray, destination: Path, title: str) -> None:
    plt = require_module("matplotlib.pyplot", "Install matplotlib to generate figures.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics = regression_metrics(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(7.4, 6.2))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#FBFCFE")
    axis.scatter(
        y_true,
        y_pred,
        s=42,
        alpha=0.78,
        color="#2878B5",
        edgecolor="white",
        linewidth=0.55,
    )
    lower = float(min(np.min(y_true), np.min(y_pred)))
    upper = float(max(np.max(y_true), np.max(y_pred)))
    padding = max(0.15, (upper - lower) * 0.04)
    limits = (lower - padding, upper + padding)
    axis.plot(
        limits,
        limits,
        linestyle="--",
        color="#374151",
        linewidth=1.35,
        label="Ideal agreement",
    )
    axis.set(xlim=limits, ylim=limits, xlabel="Actual logKp", ylabel="Predicted logKp")
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#16324F", pad=14)
    axis.text(
        0.03,
        0.97,
        f"R² = {metrics.r2:.3f}\nRMSE = {metrics.rmse:.3f}\nMAE = {metrics.mae:.3f}\nn = {len(y_true)}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        color="#16324F",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D7DEE5", "alpha": 0.94},
    )
    axis.legend(loc="lower right", frameon=False)
    axis.grid(True, color="#D7DEE5", linewidth=0.6, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def evaluate_regressor(
    estimator: Any,
    x_train,
    y_train,
    x_test,
    y_test,
    cv_folds: int,
    random_state: int,
    cv=None,
    cv_groups=None,
    cv_scoring: str = "neg_mean_absolute_error",
    cv_metric_name: str = "cv_mae",
) -> tuple[np.ndarray, dict[str, float]]:
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)
    metrics = regression_metrics(y_test, predictions).__dict__
    cv = cv or KFold(n_splits=cv_folds, shuffle=False)
    cv_scores = cross_val_score(estimator, x_train, y_train, scoring=cv_scoring, cv=cv, groups=cv_groups)
    metrics[cv_metric_name] = float(np.abs(cv_scores.mean()))
    ci_low, ci_high = bootstrap_metric_interval(
        np.asarray(y_test),
        np.asarray(predictions),
        metric_fn=lambda yt, yp: float(mean_absolute_error(yt, yp)),
        n_bootstrap=1000,
        random_state=random_state,
    )
    metrics["mae_ci_low"] = ci_low
    metrics["mae_ci_high"] = ci_high
    return np.asarray(predictions), metrics


def save_run_summary(runs: list[FittedRun], destination: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"model": run.model_name, "artifact_path": str(run.artifact_path), **run.metrics}
            for run in runs
        ]
    ).sort_values(["mae", "rmse", "r2"], ascending=[True, True, False])
    save_table(frame, destination)
    return frame


def dump_text_report(content: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def dump_json(payload: dict[str, Any], destination: Path) -> None:
    save_json(payload, destination)
