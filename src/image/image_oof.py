"""Leakage-safe OOF training for the M3 image baseline."""

from pathlib import Path

import numpy as np
import pandas as pd

from .image_cv import get_fold_split
from .image_predictor import ImagePredictor


def generate_oof_predictions(
    dataframe,
    embeddings,
    targets,
    folds,
    id_column,
    n_folds,
    alpha=1.0,
    image_valid_mask=None,
):
    """Generate leakage-safe out-of-fold predictions.

    Each fold's validation rows are predicted only by a model
    trained on the other folds.

    Missing or invalid images use a fallback fitted only on
    that fold's training targets.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float64)

    if len(dataframe) != len(embeddings):
        raise ValueError(
            "Dataframe and embeddings must contain the same number of rows."
        )

    if len(dataframe) != len(targets):
        raise ValueError(
            "Dataframe and targets must contain the same number of rows."
        )

    if dataframe[id_column].duplicated().any():
        raise ValueError("Dataframe contains duplicate IDs.")

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    if image_valid_mask is None:
        image_valid_mask = np.ones(
            len(dataframe),
            dtype=bool,
        )
    else:
        image_valid_mask = np.asarray(
            image_valid_mask,
            dtype=bool,
        )

        if len(image_valid_mask) != len(dataframe):
            raise ValueError(
                "image_valid_mask length must match dataframe."
            )

    oof_predictions = np.full(
        len(dataframe),
        np.nan,
        dtype=np.float64,
    )

    oof_seen = np.zeros(
        len(dataframe),
        dtype=bool,
    )

    id_to_index = {
        dataset_id: index
        for index, dataset_id in enumerate(dataframe[id_column])
    }

    for fold_number in range(n_folds):
        train_rows, valid_rows = get_fold_split(
            dataframe=dataframe,
            folds=folds,
            id_column=id_column,
            fold_number=fold_number,
        )

        train_ids = train_rows[id_column].tolist()
        valid_ids = valid_rows[id_column].tolist()

        train_indices = [
            id_to_index[dataset_id]
            for dataset_id in train_ids
        ]

        valid_indices = [
            id_to_index[dataset_id]
            for dataset_id in valid_ids
        ]

        predictor = ImagePredictor(alpha=alpha)

        predictor.fit(
            embeddings[train_indices],
            targets[train_indices],
            valid_mask=image_valid_mask[train_indices],
        )

        valid_predictions = predictor.predict(
            embeddings[valid_indices],
            valid_mask=image_valid_mask[valid_indices],
        )

        if np.any(oof_seen[valid_indices]):
            raise ValueError(
                f"Some IDs received OOF predictions more than once "
                f"in fold {fold_number}."
            )

        oof_predictions[valid_indices] = valid_predictions
        oof_seen[valid_indices] = True

    if not oof_seen.all():
        missing_ids = dataframe.loc[
            ~oof_seen,
            id_column,
        ].tolist()

        raise ValueError(
            f"Some IDs did not receive OOF predictions: {missing_ids}"
        )

    if not np.isfinite(oof_predictions).all():
        raise ValueError(
            "OOF predictions contain NaN or infinite values."
        )

    return pd.DataFrame(
        {
            id_column: dataframe[id_column].tolist(),
            "prediction": oof_predictions,
        }
    )


def save_oof_predictions(
    predictions,
    output_path="predictions/oof/m3_oof.csv",
):
    """Validate and save M3 OOF predictions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = ["id", "prediction"]

    if list(predictions.columns) != required_columns:
        raise ValueError(
            f"Expected columns {required_columns}, "
            f"got {list(predictions.columns)}."
        )

    if predictions["id"].duplicated().any():
        raise ValueError("OOF predictions contain duplicate IDs.")

    if predictions["id"].isna().any():
        raise ValueError("OOF predictions contain missing IDs.")

    if not np.isfinite(predictions["prediction"]).all():
        raise ValueError(
            "OOF predictions contain NaN or infinite values."
        )

    predictions.to_csv(output_path, index=False)

    return output_path