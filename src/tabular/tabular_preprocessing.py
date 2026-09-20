"""Tabular feature preparation for the shared M1 pipeline."""

from __future__ import annotations

import pandas as pd


def build_numerical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare numerical features while preserving row order."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    numeric = data.select_dtypes(include=["number"]).copy()
    numeric = numeric.replace([float("inf"), float("-inf")], pd.NA)

    for column in numeric.columns:
        if numeric[column].isna().any():
            median = numeric[column].median()
            numeric[column] = numeric[column].fillna(
                median if pd.notna(median) else 0
            )

    return numeric


def build_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    categorical = data.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).copy()

    if categorical.empty:
        return pd.DataFrame(index=data.index)

    for column in categorical.columns:
        categorical[column] = (
            categorical[column]
            .astype("string")
            .fillna("__MISSING__")
        )

    return pd.get_dummies(
        categorical,
        prefix=categorical.columns,
        prefix_sep="__",
        dtype="float32",
    )


def build_domain_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create generic text-derived tabular features."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    features = pd.DataFrame(index=data.index)

    for column in data.columns:
        series = data[column]

        if (
            pd.api.types.is_string_dtype(series)
            or series.dtype == object
        ):
            text = series.astype("string").fillna("")

            features[f"{column}__length"] = (
                text.str.len().astype("float32")
            )

            features[f"{column}__word_count"] = (
                text.str.split()
                .str.len()
                .fillna(0)
                .astype("float32")
            )

    return features


def build_tabular_features(
    data: pd.DataFrame,
    *,
    id_column: str = "id",
    target_column: str | None = None,
) -> pd.DataFrame:
    """Build the common numeric feature matrix used by M1."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    working = data.copy(deep=True)

    excluded = {id_column}

    if target_column:
        excluded.add(target_column)

    feature_data = working.drop(
        columns=[
            column
            for column in excluded
            if column in working.columns
        ]
    )

    numerical = build_numerical_features(feature_data)
    categorical = build_categorical_features(feature_data)
    domain = build_domain_features(feature_data)

    features = pd.concat(
        [numerical, categorical, domain],
        axis=1,
    )

    features = features.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    features = features.fillna(0)

    features = features.loc[
        :,
        ~features.columns.duplicated(),
    ]

    return features


def align_feature_columns(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align train and test feature matrices.

    Missing columns are filled with zero and both matrices
    receive exactly the same column order.
    """
    if not isinstance(train_features, pd.DataFrame):
        raise TypeError("train_features must be a pandas DataFrame")

    if not isinstance(test_features, pd.DataFrame):
        raise TypeError("test_features must be a pandas DataFrame")

    train = train_features.copy(deep=True)
    test = test_features.copy(deep=True)

    all_columns = train.columns.union(
        test.columns,
        sort=False,
    )

    train = train.reindex(columns=all_columns, fill_value=0)
    test = test.reindex(columns=all_columns, fill_value=0)

    train = train.fillna(0)
    test = test.fillna(0)

    return train, test
