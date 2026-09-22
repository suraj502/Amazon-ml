"""Full-data training and test prediction for the M3 image baseline."""

from pathlib import Path

import numpy as np
import pandas as pd

from .image_predictor import ImagePredictor


def generate_test_predictions(
    train_dataframe,
    test_dataframe,
    train_embeddings,
    test_embeddings,
    targets,
    id_column,
    train_image_valid_mask=None,
    test_image_valid_mask=None,
    alpha=1.0,
):
    """Train on all training data and predict the test set.

    The model is fitted only on training targets. Test targets are never
    required or used.
    """
    if not isinstance(train_dataframe, pd.DataFrame):
        raise TypeError("train_dataframe must be a pandas DataFrame.")

    if not isinstance(test_dataframe, pd.DataFrame):
        raise TypeError("test_dataframe must be a pandas DataFrame.")

    train_embeddings = np.asarray(
        train_embeddings,
        dtype=np.float32,
    )
    test_embeddings = np.asarray(
        test_embeddings,
        dtype=np.float32,
    )
    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    if id_column not in train_dataframe.columns:
        raise ValueError(
            f"ID column {id_column!r} does not exist in train_dataframe."
        )

    if id_column not in test_dataframe.columns:
        raise ValueError(
            f"ID column {id_column!r} does not exist in test_dataframe."
        )

    if train_dataframe[id_column].duplicated().any():
        raise ValueError("Training dataframe contains duplicate IDs.")

    if test_dataframe[id_column].duplicated().any():
        raise ValueError("Test dataframe contains duplicate IDs.")

    if len(train_dataframe) != len(train_embeddings):
        raise ValueError(
            "Training dataframe and embeddings must have the same length."
        )

    if len(train_dataframe) != len(targets):
        raise ValueError(
            "Training dataframe and targets must have the same length."
        )

    if len(test_dataframe) != len(test_embeddings):
        raise ValueError(
            "Test dataframe and embeddings must have the same length."
        )

    if train_image_valid_mask is None:
        train_image_valid_mask = np.ones(
            len(train_dataframe),
            dtype=bool,
        )
    else:
        train_image_valid_mask = np.asarray(
            train_image_valid_mask,
            dtype=bool,
        )

    if test_image_valid_mask is None:
        test_image_valid_mask = np.ones(
            len(test_dataframe),
            dtype=bool,
        )
    else:
        test_image_valid_mask = np.asarray(
            test_image_valid_mask,
            dtype=bool,
        )

    if len(train_image_valid_mask) != len(train_dataframe):
        raise ValueError(
            "train_image_valid_mask length must match train_dataframe."
        )

    if len(test_image_valid_mask) != len(test_dataframe):
        raise ValueError(
            "test_image_valid_mask length must match test_dataframe."
        )

    predictor = ImagePredictor(alpha=alpha)

    predictor.fit(
        train_embeddings,
        targets,
        valid_mask=train_image_valid_mask,
    )

    predictions = predictor.predict(
        test_embeddings,
        valid_mask=test_image_valid_mask,
    )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Test predictions contain NaN or infinite values."
        )

    return pd.DataFrame(
        {
            id_column: test_dataframe[id_column].tolist(),
            "prediction": predictions,
        }
    )


def save_test_predictions(
    predictions,
    output_path="predictions/test/m3_test.csv",
):
    """Validate and save M3 test predictions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = ["id", "prediction"]

    if list(predictions.columns) != required_columns:
        raise ValueError(
            f"Expected columns {required_columns}, "
            f"got {list(predictions.columns)}."
        )

    if predictions["id"].duplicated().any():
        raise ValueError("Test predictions contain duplicate IDs.")

    if predictions["id"].isna().any():
        raise ValueError("Test predictions contain missing IDs.")

    if not np.isfinite(predictions["prediction"]).all():
        raise ValueError(
            "Test predictions contain NaN or infinite values."
        )

    predictions.to_csv(output_path, index=False)

    return output_path