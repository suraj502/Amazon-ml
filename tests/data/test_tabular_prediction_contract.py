import numpy as np
import pandas as pd
import pytest

from src.tabular.predict_tabular import generate_tabular_predictions


class DummyModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, features):
        return np.asarray(self.predictions)


def test_valid_prediction_contract():
    data = pd.DataFrame({
        "id": ["A001", "A002", "A003"],
        "feature": [10, 20, 30],
    })

    model = DummyModel([1.1, 2.2, 3.3])

    result = generate_tabular_predictions(data, model)

    assert list(result.columns) == ["id", "prediction"]
    assert result["id"].tolist() == ["A001", "A002", "A003"]
    assert result["prediction"].tolist() == [1.1, 2.2, 3.3]
    assert len(result) == len(data)


def test_duplicate_ids_are_rejected():
    data = pd.DataFrame({
        "id": ["A001", "A001"],
        "feature": [10, 20],
    })

    model = DummyModel([1.0, 2.0])

    with pytest.raises(ValueError, match="Duplicate IDs"):
        generate_tabular_predictions(data, model)


def test_missing_ids_are_rejected():
    data = pd.DataFrame({
        "id": ["A001", None],
        "feature": [10, 20],
    })

    model = DummyModel([1.0, 2.0])

    with pytest.raises(ValueError, match="missing values"):
        generate_tabular_predictions(data, model)


def test_empty_ids_are_rejected():
    data = pd.DataFrame({
        "id": ["A001", "   "],
        "feature": [10, 20],
    })

    model = DummyModel([1.0, 2.0])

    with pytest.raises(ValueError, match="empty values"):
        generate_tabular_predictions(data, model)


def test_nan_predictions_are_rejected():
    data = pd.DataFrame({
        "id": ["A001", "A002"],
        "feature": [10, 20],
    })

    model = DummyModel([1.0, np.nan])

    with pytest.raises(ValueError, match="NaN or infinite"):
        generate_tabular_predictions(data, model)


def test_infinite_predictions_are_rejected():
    data = pd.DataFrame({
        "id": ["A001", "A002"],
        "feature": [10, 20],
    })

    model = DummyModel([1.0, np.inf])

    with pytest.raises(ValueError, match="NaN or infinite"):
        generate_tabular_predictions(data, model)


def test_prediction_count_must_match_rows():
    data = pd.DataFrame({
        "id": ["A001", "A002", "A003"],
        "feature": [10, 20, 30],
    })

    model = DummyModel([1.0, 2.0])

    with pytest.raises(ValueError, match="Prediction count"):
        generate_tabular_predictions(data, model)


def test_missing_id_column_is_rejected():
    data = pd.DataFrame({
        "feature": [10, 20],
    })

    model = DummyModel([1.0, 2.0])

    with pytest.raises(ValueError, match="Missing required ID column"):
        generate_tabular_predictions(data, model)


def test_ids_are_preserved_in_original_order():
    data = pd.DataFrame({
        "id": ["ID-C", "ID-A", "ID-B"],
        "feature": [30, 10, 20],
    })

    model = DummyModel([300.0, 100.0, 200.0])

    result = generate_tabular_predictions(data, model)

    assert result["id"].tolist() == ["ID-C", "ID-A", "ID-B"]
    assert result["prediction"].tolist() == [300.0, 100.0, 200.0]
