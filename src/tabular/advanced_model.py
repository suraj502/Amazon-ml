"""Tree-based tabular models for the shared M1 pipeline."""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
import xgboost as xgb


def train_advanced_model(
    features,
    targets,
    *,
    model_type="lightgbm",
    random_state=42,
):
    """Train a LightGBM or XGBoost regression model."""
    if features is None or targets is None:
        raise ValueError("features and targets are required")

    if len(features) != len(targets):
        raise ValueError("features and targets must contain the same number of rows")

    if len(features) == 0:
        raise ValueError("Cannot train on empty data")

    if model_type == "lightgbm":
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "Expected 'lightgbm' or 'xgboost'."
        )

    model.fit(features, targets)
    return model


def predict_advanced_model(model, features):
    """Generate predictions from a trained tree-based model."""
    if model is None:
        raise ValueError("model is required")

    if features is None:
        raise ValueError("features are required")

    predictions = np.asarray(model.predict(features), dtype=float)

    if not np.isfinite(predictions).all():
        raise ValueError("Advanced model produced non-finite predictions")

    return predictions
