"""Canonical preprocessing for the shared M1 data pipeline."""

import pandas as pd

from src.data.validation import DataValidationError


def clean_data(
    data: pd.DataFrame,
    *,
    id_column="id",
    id_as_string=True,
    strip_header_whitespace=True,
):
    """
    Create the canonical in-memory representation of a dataset.

    This function intentionally performs only contract-level
    transformations. It does not impute values, remove rows,
    lowercase feature values, or alter target values.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    cleaned = data.copy(deep=True)

    if strip_header_whitespace:
        cleaned.columns = [
            column.strip() if isinstance(column, str) else column
            for column in cleaned.columns
        ]

    duplicate_columns = cleaned.columns[
        cleaned.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        raise DataValidationError(
            f"Duplicate columns found after preprocessing: "
            f"{duplicate_columns}"
        )

    if id_column in cleaned.columns and id_as_string:
        if cleaned[id_column].isna().any():
            raise DataValidationError(
                f"ID column '{id_column}' contains missing values"
            )

        cleaned[id_column] = cleaned[id_column].astype(str)

        if cleaned[id_column].str.strip().eq("").any():
            raise DataValidationError(
                f"ID column '{id_column}' contains empty values"
            )

    return cleaned


def clean_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    id_column="id",
    id_as_string=True,
    strip_header_whitespace=True,
):
    """Apply identical canonical preprocessing to train and test."""
    cleaned_train = clean_data(
        train,
        id_column=id_column,
        id_as_string=id_as_string,
        strip_header_whitespace=strip_header_whitespace,
    )

    cleaned_test = clean_data(
        test,
        id_column=id_column,
        id_as_string=id_as_string,
        strip_header_whitespace=strip_header_whitespace,
    )

    return cleaned_train, cleaned_test
