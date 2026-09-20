"""Prediction orchestration for the M3 image baseline."""

import numpy as np

from .image_fallback import RegressionFallback
from .image_model import ImageRegressionModel


class ImagePredictor:
    """Combine the image model with a deterministic fallback."""

    def __init__(self, alpha=1.0):
        self.model = ImageRegressionModel(alpha=alpha)
        self.fallback = RegressionFallback()
        self.is_fitted = False

    def fit(self, embeddings, targets, valid_mask=None):
        """Fit the image model and fallback using training data."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float64)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")

        if targets.ndim != 1:
            raise ValueError("targets must be a 1D array.")

        if len(embeddings) != len(targets):
            raise ValueError(
                "Number of embeddings and targets must match."
            )

        self.fallback.fit(targets)

        if valid_mask is None:
            valid_mask = np.ones(len(targets), dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)

            if len(valid_mask) != len(targets):
                raise ValueError(
                    "valid_mask length must match targets."
                )

        if not valid_mask.any():
            raise ValueError(
                "At least one valid image is required to fit the "
                "image model."
            )

        self.model.fit(
            embeddings[valid_mask],
            targets[valid_mask],
        )

        self.is_fitted = True

        return self

    def predict(self, embeddings, valid_mask=None):
        """Generate predictions with fallback for unavailable images."""
        if not self.is_fitted:
            raise RuntimeError(
                "ImagePredictor must be fitted before prediction."
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")

        count = len(embeddings)

        if valid_mask is None:
            valid_mask = np.ones(count, dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)

            if len(valid_mask) != count:
                raise ValueError(
                    "valid_mask length must match embeddings."
                )

        predictions = self.fallback.predict(count)

        if valid_mask.any():
            predictions[valid_mask] = self.model.predict(
                embeddings[valid_mask]
            )

        if not np.isfinite(predictions).all():
            raise ValueError(
                "Predictions contain NaN or infinite values."
            )

        return predictions