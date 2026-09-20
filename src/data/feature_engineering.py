"""Shared and tabular feature engineering for the M1 data pipeline."""

import pandas as pd


def _find_duplicate_columns(data: pd.DataFrame) -> list[str]:
    """Return duplicate column names."""
    return data.columns[data.columns.duplicated()].tolist()


def _find_constant_columns(data: pd.DataFrame) -> list[str]:
    """Return columns containing at most one unique non-null value."""
    constant_columns = []

    for column in data.columns:
        if data[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)

    return constant_columns


def _add_missing_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add binary indicators for columns containing missing values."""
    result = data.copy()

    for column in list(data.columns):
        if data[column].isna().any():
            indicator_name = f"{column}__missing"

            # Avoid overwriting an existing user column.
            if indicator_name not in result.columns:
                result[indicator_name] = data[column].isna().astype("int8")

    return result


def _add_datetime_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add safe calendar features for columns already recognized as datetimes.

    We intentionally do not parse arbitrary object/string columns here.
    That prevents accidental interpretation of IDs or text as dates.
    """
    result = data.copy()

    datetime_columns = result.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns

    for column in datetime_columns:
        prefix = f"{column}__"

        result[f"{prefix}year"] = result[column].dt.year
        result[f"{prefix}month"] = result[column].dt.month
        result[f"{prefix}day"] = result[column].dt.day
        result[f"{prefix}dayofweek"] = result[column].dt.dayofweek

    return result


def build_features(
    data: pd.DataFrame,
    *,
    add_missing_indicators=True,
    add_datetime_features=True,
    remove_constant_features=True,
    remove_duplicate_columns=True,
) -> pd.DataFrame:
    """
    Build shared, leakage-safe M1 features.

    This function performs only generic transformations that are safe
    to share with the other pipeline members.

    It does not:
    - use the target
    - perform target encoding
    - create text embeddings
    - create image features
    - drop rows
    - impute missing values
    - parse arbitrary text as dates
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    features = data.copy(deep=True)

    if remove_duplicate_columns:
        duplicate_columns = _find_duplicate_columns(features)

        if duplicate_columns:
            features = features.loc[
                :, ~features.columns.duplicated()
            ]

    if remove_constant_features:
        constant_columns = _find_constant_columns(features)

        if constant_columns:
            features = features.drop(
                columns=constant_columns
            )

    if add_missing_indicators:
        features = _add_missing_indicators(features)

    if add_datetime_features:
        features = _add_datetime_features(features)

    return features
