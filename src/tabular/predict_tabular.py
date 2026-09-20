"""Prediction generation for the shared M1 tabular pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_tabular_predictions(
    data: pd.DataFrame,
    model,
    *,
    id_column: str = "id",
) -> pd.DataFrame:
    """
    Generate predictions in the common ``id,prediction`` format.

    IDs are preserved exactly in their input order.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if id_column not in data.columns:
        raise ValueError(f"Missing required ID column: {id_column}")

    ids = data[id_column]

    if ids.isna().any():
        raise ValueError(f"ID column '{id_column}' contains missing values")

    ids = ids.astype(str)

    if ids.str.strip().eq("").any():
        raise ValueError(f"ID column '{id_column}' contains empty values")

    if ids.duplicated().any():
        duplicates = ids[ids.duplicated()].unique().tolist()
        raise ValueError(
            f"Duplicate IDs found in '{id_column}': {duplicates}"
        )

    feature_data = data.drop(columns=[id_column])

    predictions = model.predict(feature_data)

    predictions = np.asarray(predictions).reshape(-1)

    if len(predictions) != len(data):
        raise ValueError(
            "Prediction count does not match input row count: "
            f"{len(predictions)} != {len(data)}"
        )

    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN or infinite values")

    return pd.DataFrame(
        {
            id_column: ids,
            "prediction": predictions,
        }
    )
