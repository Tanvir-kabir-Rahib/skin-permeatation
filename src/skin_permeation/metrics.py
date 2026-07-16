from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dependencies import require_module

sk_metrics = require_module("sklearn.metrics", "Install scikit-learn to compute regression metrics.")

r2_score = sk_metrics.r2_score
mean_absolute_error = sk_metrics.mean_absolute_error
mean_squared_error = sk_metrics.mean_squared_error


@dataclass(frozen=True)
class RegressionMetrics:
    r2: float
    rmse: float
    mae: float


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    return RegressionMetrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
    )


def bootstrap_metric_interval(
    y_true,
    y_pred,
    metric_fn,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(y_true))
    estimates = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        estimates.append(metric_fn(np.asarray(y_true)[sampled], np.asarray(y_pred)[sampled]))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))
