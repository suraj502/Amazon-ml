"""Validation utilities for M3 image prediction artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd


def validate_prediction_file(
    predictions: pd.DataFrame,
    expected_ids,
    id_column: str = "id",
) -> None:
    """Validate an M3 prediction dataframe against expected IDs."""

    expected_ids = list(expected_ids)

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame.")

    required_columns = [id_column, "prediction"]

    if predictions.columns.tolist() != required_columns:
        raise ValueError(
            f"Prediction columns must be exactly {required_columns}, "
            f"got {predictions.columns.tolist()}."
        )

    if predictions[id_column].isna().any():
        raise ValueError("Prediction file contains null IDs.")

    if predictions[id_column].duplicated().any():
        raise ValueError("Prediction file contains duplicate IDs.")

    actual_ids = predictions[id_column].tolist()

    if set(actual_ids) != set(expected_ids):
        missing_ids = sorted(set(expected_ids) - set(actual_ids))
        extra_ids = sorted(set(actual_ids) - set(expected_ids))

        raise ValueError(
            f"Prediction ID mismatch. "
            f"missing={missing_ids}, extra={extra_ids}"
        )

    if not np.isfinite(predictions["prediction"].to_numpy()).all():
        raise ValueError("Prediction file contains NaN or infinite values.")

    if len(predictions) != len(expected_ids):
        raise ValueError(
            f"Prediction row count mismatch: "
            f"expected {len(expected_ids)}, got {len(predictions)}."
        )


def validate_prediction_csv(
    path,
    expected_ids,
    id_column: str = "id",
) -> pd.DataFrame:
    """Load and validate a prediction CSV."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {path}")

    predictions = pd.read_csv(path)

    validate_prediction_file(
        predictions=predictions,
        expected_ids=expected_ids,
        id_column=id_column,
    )

    return predictions