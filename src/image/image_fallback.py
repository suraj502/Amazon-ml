"""Deterministic fallback predictions for unavailable images."""

import numpy as np


class RegressionFallback:
    """Use the training-target mean for unavailable images."""

    def __init__(self):
        self.value = None

    def fit(self, targets):
        """Fit the fallback value using training targets."""
        targets = np.asarray(targets, dtype=np.float64)

        if targets.ndim != 1:
            raise ValueError("targets must be a 1D array.")

        if len(targets) == 0:
            raise ValueError("targets cannot be empty.")

        if not np.isfinite(targets).all():
            raise ValueError(
                "targets contain NaN or infinite values."
            )

        self.value = float(np.mean(targets))

        return self

    def predict(self, count):
        """Return the fallback prediction for a number of rows."""
        if self.value is None:
            raise RuntimeError(
                "RegressionFallback must be fitted before prediction."
            )

        if count < 0:
            raise ValueError("count cannot be negative.")

        return np.full(
            count,
            self.value,
            dtype=np.float64,
        )