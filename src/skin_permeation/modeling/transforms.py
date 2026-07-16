from __future__ import annotations

import numpy as np
import pandas as pd

from ..dependencies import require_module

sk_base = require_module("sklearn.base", "Install scikit-learn to use feature transformers.")

BaseEstimator = sk_base.BaseEstimator
TransformerMixin = sk_base.TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one side of highly correlated feature pairs using training data only."""

    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold

    def fit(self, X, y=None):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        corr = frame.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.feature_names_in_ = list(frame.columns)
        self.columns_to_drop_ = [column for column in upper.columns if any(upper[column] >= self.threshold)]
        self.feature_names_out_ = [column for column in self.feature_names_in_ if column not in self.columns_to_drop_]
        self.keep_indices_ = [self.feature_names_in_.index(column) for column in self.feature_names_out_]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.feature_names_out_]
        return np.asarray(X)[:, self.keep_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)
