"""Lightweight prediction models for M3 image embeddings."""

import numpy as np
from sklearn.linear_model import Ridge


class ImageRegressionModel:
    """Simple Ridge regression head for frozen image embeddings."""

    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")

        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, embeddings, targets):
        """Fit the regression head."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        targets = np.asarray(targets)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")

        if len(embeddings) != len(targets):
            raise ValueError(
                "Number of embeddings and targets must match."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings contain NaN or infinite values.")

        if not np.isfinite(targets).all():
            raise ValueError("targets contain NaN or infinite values.")

        self.model.fit(embeddings, targets)

        return self

    def predict(self, embeddings):
        """Generate predictions from image embeddings."""
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")

        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings contain NaN or infinite values.")

        predictions = self.model.predict(embeddings)

        return np.asarray(predictions, dtype=np.float64)