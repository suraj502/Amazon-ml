"""Generic, competition-safe data validation for the shared M1 pipeline."""

from pathlib import Path
from typing import Iterable

import pandas as pd


class DataValidationError(ValueError):
    """Raised when the dataset violates the shared data contract."""


def load_csv(path, encoding="utf-8-sig"):
    """Load a CSV dataset without modifying the source file."""
    path = Path(path)

    if not path.exists():
        raise DataValidationError(f"Dataset not found: {path}")

    if not path.is_file():
        raise DataValidationError(f"Dataset path is not a file: {path}")

    try:
        return pd.read_csv(path, encoding=encoding)
    except Exception as exc:
        raise DataValidationError(
            f"Unable to read dataset '{path}': {exc}"
        ) from exc


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """Ensure all required columns exist exactly once."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    columns = list(data.columns)
    required = list(required_columns)

    duplicates = data.columns[data.columns.duplicated()].tolist()
    if duplicates:
        raise DataValidationError(
            f"Duplicate columns found: {duplicates}"
        )

    missing = [column for column in required if column not in columns]

    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}"
        )


def validate_id_column(
    data: pd.DataFrame,
    id_column: str = "id",
) -> None:
    """Validate the shared ID contract."""
    validate_required_columns(data, [id_column])

    ids = data[id_column]

    if ids.isna().any():
        raise DataValidationError(
            f"ID column '{id_column}' contains missing values"
        )

    if ids.astype(str).str.strip().eq("").any():
        raise DataValidationError(
            f"ID column '{id_column}' contains empty values"
        )

    if ids.duplicated().any():
        duplicates = ids[ids.duplicated()].astype(str).unique().tolist()
        raise DataValidationError(
            f"Duplicate IDs found in '{id_column}': {duplicates}"
        )


def validate_train_test_ids(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str = "id",
) -> None:
    """Ensure train/test IDs are valid and do not overlap."""
    validate_id_column(train, id_column)
    validate_id_column(test, id_column)

    overlap = set(train[id_column]).intersection(set(test[id_column]))

    if overlap:
        raise DataValidationError(
            f"Train/test ID overlap detected: {list(overlap)[:10]}"
        )


def validate_target(
    train: pd.DataFrame,
    target: str,
) -> None:
    """Validate the target column on training data."""
    validate_required_columns(train, [target])

    target_values = train[target]

    if target_values.isna().any():
        raise DataValidationError(
            f"Target column '{target}' contains missing values"
        )

    if target_values.isnull().any():
        raise DataValidationError(
            f"Target column '{target}' contains null values"
        )

    if target_values.nunique(dropna=True) == 0:
        raise DataValidationError(
            f"Target column '{target}' contains no usable values"
        )


def validate_dataset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    id_column: str = "id",
    target: str | None = None,
) -> dict:
    """
    Run the complete shared data contract validation.

    Returns a compact validation report when all checks pass.
    """
    if not isinstance(train, pd.DataFrame):
        raise TypeError("train must be a pandas DataFrame")

    if not isinstance(test, pd.DataFrame):
        raise TypeError("test must be a pandas DataFrame")

    validate_id_column(train, id_column)
    validate_id_column(test, id_column)
    validate_train_test_ids(train, test, id_column)

    if target:
        validate_target(train, target)

        if target in test.columns:
            raise DataValidationError(
                f"Target column '{target}' must not exist in test data"
            )

    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_columns": len(train.columns),
        "test_columns": len(test.columns),
        "id_column": id_column,
        "target": target,
        "status": "valid",
    }
