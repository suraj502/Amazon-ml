"""Shared-CV adapter for the M3 image pipeline."""

from pathlib import Path

import pandas as pd
import yaml

from src.utils.cv_split import validate_folds


def load_validation_config(config_path="config/config.yaml"):
    """Load the shared validation configuration."""
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["validation"]


def load_shared_folds(
    dataframe,
    id_column,
    config_path="config/config.yaml",
):
    """Load and validate the shared competition folds.

    Returns:
        pandas.DataFrame: Original IDs with their shared fold assignments.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if id_column not in dataframe.columns:
        raise ValueError(
            f"ID column {id_column!r} does not exist in the dataframe."
        )

    validation_config = load_validation_config(config_path)

    folds_path = validation_config["folds_path"]
    n_folds = validation_config["n_folds"]
    warn_small_fold_size = validation_config.get("warn_small_fold_size")

    expected_ids = dataframe[id_column].tolist()

    fold_mapping = validate_folds(
        path=folds_path,
        expected_ids=expected_ids,
        n_folds=n_folds,
        warn_small_fold_size=warn_small_fold_size,
    )

    folds = pd.DataFrame(
        {
            id_column: expected_ids,
            "fold": [
                fold_mapping[dataset_id]
                for dataset_id in expected_ids
            ],
        }
    )

    return folds


def get_fold_split(
    dataframe,
    folds,
    id_column,
    fold_number,
):
    """Return train and validation rows for one shared fold."""
    if fold_number < 0:
        raise ValueError("fold_number must be non-negative.")

    if id_column not in dataframe.columns:
        raise ValueError(
            f"ID column {id_column!r} does not exist in the dataframe."
        )

    if id_column not in folds.columns:
        raise ValueError(
            f"ID column {id_column!r} does not exist in the folds."
        )

    if "fold" not in folds.columns:
        raise ValueError("Shared folds must contain a 'fold' column.")

    fold_lookup = folds[[id_column, "fold"]].copy()

    merged = dataframe.merge(
        fold_lookup,
        on=id_column,
        how="left",
        sort=False,
        validate="one_to_one",
    )

    if merged["fold"].isna().any():
        raise ValueError(
            "Some dataframe IDs do not have a fold assignment."
        )

    train = merged[merged["fold"] != fold_number].copy()
    valid = merged[merged["fold"] == fold_number].copy()

    return train, valid