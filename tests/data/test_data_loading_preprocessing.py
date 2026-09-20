import pandas as pd
import pytest

from src.data.data_loader import (
    load_raw_data,
    load_train_test_data,
    require_columns,
)
from src.data.preprocessing import (
    clean_data,
    clean_train_test,
)
from src.data.validation import DataValidationError


def test_load_raw_data_strips_header_whitespace(tmp_path):
    path = tmp_path / "train.csv"

    path.write_text(
        " id , price ,name \n1,10,Phone\n",
        encoding="utf-8",
    )

    data = load_raw_data(path)

    assert list(data.columns) == ["id", "price", "name"]


def test_load_raw_data_missing_file_fails(tmp_path):
    path = tmp_path / "missing.csv"

    with pytest.raises(DataValidationError, match="Dataset not found"):
        load_raw_data(path)


def test_load_raw_data_duplicate_headers_fail(tmp_path):
    path = tmp_path / "duplicate.csv"

    path.write_text(
        " id ,id,name\n1,2,Phone\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="Duplicate columns"):
        load_raw_data(path)


def test_load_train_test_data(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    train_path.write_text(
        "id,target\n1,10\n2,20\n",
        encoding="utf-8",
    )

    test_path.write_text(
        "id,name\n3,Phone\n",
        encoding="utf-8",
    )

    train, test = load_train_test_data(
        train_path,
        test_path,
    )

    assert len(train) == 2
    assert len(test) == 1
    assert list(train.columns) == ["id", "target"]
    assert list(test.columns) == ["id", "name"]


def test_require_columns():
    data = pd.DataFrame(
        {
            "id": ["1"],
            "target": [10],
        }
    )

    result = require_columns(
        data,
        ["id", "target"],
    )

    assert result is data


def test_clean_data_converts_id_to_string():
    data = pd.DataFrame(
        {
            "id": [1001, 1002],
            "price": [10.5, 20.5],
        }
    )

    cleaned = clean_data(data)

    assert pd.api.types.is_string_dtype(cleaned["id"])
    assert cleaned["id"].tolist() == ["1001", "1002"]


def test_clean_data_preserves_feature_values():
    data = pd.DataFrame(
        {
            "id": ["A1"],
            "name": ["Premium Phone"],
            "price": [999.5],
            "description": [None],
        }
    )

    cleaned = clean_data(data)

    assert cleaned.loc[0, "name"] == "Premium Phone"
    assert cleaned.loc[0, "price"] == 999.5
    assert pd.isna(cleaned.loc[0, "description"])


def test_clean_data_does_not_modify_original():
    data = pd.DataFrame(
        {
            " id ": [1001],
            "price": [100],
        }
    )

    original = data.copy(deep=True)

    clean_data(data)

    pd.testing.assert_frame_equal(data, original)


def test_clean_data_missing_id_fails():
    data = pd.DataFrame(
        {
            "id": [None],
            "price": [100],
        }
    )

    with pytest.raises(DataValidationError, match="missing values"):
        clean_data(data)


def test_clean_data_empty_id_fails():
    data = pd.DataFrame(
        {
            "id": ["   "],
            "price": [100],
        }
    )

    with pytest.raises(DataValidationError, match="empty values"):
        clean_data(data)


def test_clean_train_test_uses_same_contract():
    train = pd.DataFrame(
        {
            " id ": [1, 2],
            "target": [10, 20],
        }
    )

    test = pd.DataFrame(
        {
            " id ": [3],
            "name": ["Phone"],
        }
    )

    cleaned_train, cleaned_test = clean_train_test(
        train,
        test,
    )

    assert list(cleaned_train.columns) == ["id", "target"]
    assert list(cleaned_test.columns) == ["id", "name"]
    assert cleaned_train["id"].tolist() == ["1", "2"]
    assert cleaned_test["id"].tolist() == ["3"]
