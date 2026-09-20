"""Baseline tabular model for the shared M1 pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor


def train_baseline(features, targets):
    """
    Train a deterministic median baseline.

    This intentionally provides a simple reference point before
    comparing stronger tree-based models.
    """
    if features is None or targets is None:
        raise ValueError("features and targets are required")

    if len(features) != len(targets):
        raise ValueError("features and targets must contain the same number of rows")

    if len(features) == 0:
        raise ValueError("Cannot train baseline on empty data")

    model = DummyRegressor(strategy="median")
    model.fit(features, targets)

    return model


def predict_baseline(model, features):
    """Generate baseline predictions."""
    if model is None:
        raise ValueError("model is required")

    if features is None:
        raise ValueError("features are required")

    predictions = model.predict(features)

    predictions = np.asarray(predictions, dtype=float)

    if not np.isfinite(predictions).all():
        raise ValueError("Baseline model produced non-finite predictions")

    return predictions
