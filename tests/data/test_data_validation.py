import pandas as pd
import pytest

from src.data.validation import (
    DataValidationError,
    validate_dataset,
    validate_id_column,
    validate_required_columns,
)


def test_required_columns_pass():
    df = pd.DataFrame({
        "id": ["1", "2"],
        "price": [10, 20],
    })

    validate_required_columns(df, ["id", "price"])


def test_required_columns_fail():
    df = pd.DataFrame({"id": ["1", "2"]})

    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_required_columns(df, ["id", "price"])


def test_duplicate_ids_fail():
    df = pd.DataFrame({
        "id": ["1", "1"],
        "price": [10, 20],
    })

    with pytest.raises(DataValidationError, match="Duplicate IDs"):
        validate_id_column(df, "id")


def test_missing_ids_fail():
    df = pd.DataFrame({
        "id": ["1", None],
        "price": [10, 20],
    })

    with pytest.raises(DataValidationError, match="missing values"):
        validate_id_column(df, "id")


def test_train_test_overlap_fails():
    train = pd.DataFrame({
        "id": ["1", "2"],
        "target": [10, 20],
    })

    test = pd.DataFrame({
        "id": ["2", "3"],
    })

    with pytest.raises(DataValidationError, match="overlap"):
        validate_dataset(
            train,
            test,
            id_column="id",
            target="target",
        )


def test_valid_dataset():
    train = pd.DataFrame({
        "id": ["1", "2", "3"],
        "target": [10.0, 20.0, 30.0],
    })

    test = pd.DataFrame({
        "id": ["4", "5"],
    })

    report = validate_dataset(
        train,
        test,
        id_column="id",
        target="target",
    )

    assert report["status"] == "valid"
    assert report["train_rows"] == 3
    assert report["test_rows"] == 2


def test_target_missing_fails():
    train = pd.DataFrame({
        "id": ["1", "2"],
        "target": [10.0, None],
    })

    test = pd.DataFrame({
        "id": ["3"],
    })

    with pytest.raises(DataValidationError, match="Target column"):
        validate_dataset(
            train,
            test,
            id_column="id",
            target="target",
        )


def test_target_in_test_fails():
    train = pd.DataFrame({
        "id": ["1", "2"],
        "target": [10.0, 20.0],
    })

    test = pd.DataFrame({
        "id": ["3"],
        "target": [30.0],
    })

    with pytest.raises(DataValidationError, match="must not exist"):
        validate_dataset(
            train,
            test,
            id_column="id",
            target="target",
        )
