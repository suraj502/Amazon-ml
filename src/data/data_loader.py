"""Data loading utilities for the shared M1 data pipeline."""

from pathlib import Path

import pandas as pd

from src.data.validation import (
    DataValidationError,
    validate_required_columns,
)


def load_raw_data(
    path,
    *,
    encoding="utf-8-sig",
    strip_header_whitespace=True,
):
    """
    Load a CSV file without modifying the source file.

    The returned DataFrame is a new in-memory representation.
    """
    path = Path(path)

    if not path.exists():
        raise DataValidationError(f"Dataset not found: {path}")

    if not path.is_file():
        raise DataValidationError(f"Dataset path is not a file: {path}")

    try:
        data = pd.read_csv(path, encoding=encoding)
    except Exception as exc:
        raise DataValidationError(
            f"Unable to read dataset '{path}': {exc}"
        ) from exc

    if strip_header_whitespace:
        data.columns = [
            column.strip() if isinstance(column, str) else column
            for column in data.columns
        ]

    duplicate_columns = data.columns[
        data.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        raise DataValidationError(
            f"Duplicate columns found after header normalization: "
            f"{duplicate_columns}"
        )

    return data


def load_train_test_data(
    train_path,
    test_path,
    *,
    encoding="utf-8-sig",
    strip_header_whitespace=True,
):
    """Load train and test CSV files using the same loading contract."""
    train = load_raw_data(
        train_path,
        encoding=encoding,
        strip_header_whitespace=strip_header_whitespace,
    )

    test = load_raw_data(
        test_path,
        encoding=encoding,
        strip_header_whitespace=strip_header_whitespace,
    )

    return train, test


def require_columns(data, required_columns):
    """Raise a validation error when required columns are missing."""
    validate_required_columns(data, required_columns)
    return data
