from __future__ import annotations

import numpy as np
import pandas as pd

from ..dependencies import require_module

NearestNeighbors = require_module("sklearn.neighbors", "Install scikit-learn to run applicability-domain analysis.").NearestNeighbors


def knn_applicability_domain(
    x_train: pd.DataFrame,
    x_query: pd.DataFrame,
    k: int = 5,
    threshold_quantile: float = 0.95,
) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=min(k, len(x_train)))
    nn.fit(x_train)
    train_distances = nn.kneighbors(x_train, return_distance=True)[0].mean(axis=1)
    query_distances = nn.kneighbors(x_query, return_distance=True)[0].mean(axis=1)
    threshold = float(np.quantile(train_distances, threshold_quantile))
    return pd.DataFrame(
        {
            "mean_knn_distance": query_distances,
            "applicability_threshold": threshold,
            "within_domain": query_distances <= threshold,
        }
    )
