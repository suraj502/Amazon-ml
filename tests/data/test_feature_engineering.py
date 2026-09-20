import pandas as pd
import pytest

from src.data.feature_engineering import build_features


def test_build_features_returns_dataframe():
    data = pd.DataFrame({
        "id": ["1", "2"],
        "price": [10, 20],
    })

    features = build_features(data)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 2


def test_missing_indicator_is_created():
    data = pd.DataFrame({
        "id": ["1", "2"],
        "price": [10, None],
    })

    features = build_features(data)

    assert "price__missing" in features.columns
    assert features["price__missing"].tolist() == [0, 1]


def test_no_missing_indicator_for_complete_column():
    data = pd.DataFrame({
        "id": ["1", "2"],
        "price": [10, 20],
    })

    features = build_features(data)

    assert "price__missing" not in features.columns


def test_constant_features_are_removed():
    data = pd.DataFrame({
        "id": ["1", "2", "3"],
        "constant": [5, 5, 5],
        "price": [10, 20, 30],
    })

    features = build_features(data)

    assert "constant" not in features.columns
    assert "price" in features.columns


def test_constant_missing_only_column_is_removed():
    data = pd.DataFrame({
        "id": ["1", "2"],
        "empty": [None, None],
        "price": [10, 20],
    })

    features = build_features(data)

    assert "empty" not in features.columns


def test_duplicate_columns_are_removed():
    data = pd.DataFrame(
        [
            ["1", 10, 100],
            ["2", 20, 200],
        ],
        columns=["id", "price", "price"],
    )

    features = build_features(data)

    assert list(features.columns) == ["id", "price"]


def test_datetime_features_are_created():
    data = pd.DataFrame({
        "id": ["1", "2"],
        "created_at": pd.to_datetime([
            "2026-01-15",
            "2026-02-20",
        ]),
    })

    features = build_features(data)

    assert "created_at__year" in features.columns
    assert "created_at__month" in features.columns
    assert "created_at__day" in features.columns
    assert "created_at__dayofweek" in features.columns

    assert features["created_at__year"].tolist() == [2026, 2026]
    assert features["created_at__month"].tolist() == [1, 2]


def test_string_dates_are_not_automatically_parsed():
    data = pd.DataFrame({
        "id": ["1", "2"],
        "date_text": ["2026-01-15", "2026-02-20"],
    })

    features = build_features(data)

    assert "date_text__year" not in features.columns


def test_original_data_is_not_modified():
    data = pd.DataFrame({
        "id": ["1", "2"],
        "price": [10, None],
        "constant": [5, 5],
    })

    original = data.copy(deep=True)

    build_features(data)

    pd.testing.assert_frame_equal(data, original)


def test_target_is_not_required():
    data = pd.DataFrame({
        "id": ["1", "2"],
        "price": [10, 20],
    })

    features = build_features(data)

    assert "id" in features.columns
    assert "price" in features.columns


def test_features_can_be_disabled():
    data = pd.DataFrame({
        "id": ["1", "2"],
        "price": [10, None],
        "constant": [5, 5],
    })

    features = build_features(
        data,
        add_missing_indicators=False,
        add_datetime_features=False,
        remove_constant_features=False,
    )

    assert list(features.columns) == [
        "id",
        "price",
        "constant",
    ]

def test_id_is_preserved_when_constant_feature_removal_is_enabled():
    data = pd.DataFrame({
        "id": ["1", "1"],
        "price": [10, 20],
    })

    features = build_features(data)

    assert "id" in features.columns
    assert features["id"].tolist() == ["1", "1"]
