import csv
import math
import tempfile
import unittest
import warnings
from pathlib import Path

from src.utils.config import ConfigurationError, validate_config
from src.utils.cv_split import FoldValidationError, validate_folds
from src.utils.metrics import evaluate_predictions
from src.utils.prediction_contract import (
    PredictionValidationError,
    align_predictions,
    load_configured_predictions,
    load_prediction_file,
)


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "m4"
REGRESSION_TASK = {"type": "regression", "id_column": "id"}
BINARY_TASK = {"type": "binary_classification", "id_column": "id"}
MULTICLASS_TASK = {
    "type": "multiclass_classification",
    "id_column": "id",
    "classes": ["cat", "dog"],
}


class Phase1ContractTests(unittest.TestCase):
    def write_csv(self, directory, name, rows):
        path = Path(directory) / name
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerows(rows)
        return path

    def test_valid_prediction_schema_and_id_normalization(self):
        table = load_prediction_file(FIXTURE_DIR / "m1_oof.csv", REGRESSION_TASK, ["001", "002", "003"])
        self.assertEqual(table.ids, ("003", "001", "002"))
        self.assertEqual(table.columns, ("prediction",))

    def test_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, "duplicate.csv", [["id", "prediction"], ["001", "1"], ["001", "2"]])
            with self.assertRaises(PredictionValidationError):
                load_prediction_file(path, REGRESSION_TASK)

    def test_missing_and_extra_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, "mismatch.csv", [["id", "prediction"], ["001", "1"], ["004", "4"]])
            with self.assertRaises(PredictionValidationError):
                load_prediction_file(path, REGRESSION_TASK, ["001", "002"])

    def test_nan_and_infinite_predictions(self):
        for value in ("nan", "inf"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = self.write_csv(directory, "invalid.csv", [["id", "prediction"], ["001", value]])
                with self.assertRaises(PredictionValidationError):
                    load_prediction_file(path, REGRESSION_TASK)

    def test_invalid_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, "columns.csv", [["id", "score"], ["001", "1"]])
            with self.assertRaises(PredictionValidationError):
                load_prediction_file(path, REGRESSION_TASK)

    def test_invalid_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, "probability.csv", [["id", "prediction"], ["001", "1.1"]])
            with self.assertRaises(PredictionValidationError):
                load_prediction_file(path, BINARY_TASK)

    def test_constant_predictions_warn(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, "constant.csv", [["id", "prediction"], ["001", "1"], ["002", "1"]])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_prediction_file(path, REGRESSION_TASK)
            self.assertTrue(any("Constant predictions" in str(item.message) for item in caught))

    def test_multiclass_columns_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = self.write_csv(directory, "valid.csv", [["id", "prob_cat", "prob_dog"], ["001", "0.25", "0.75"]])
            table = load_prediction_file(valid, MULTICLASS_TASK)
            self.assertEqual(table.columns, ("prob_cat", "prob_dog"))
            wrong = self.write_csv(directory, "wrong.csv", [["id", "prob_dog", "prob_cat"], ["001", "0.75", "0.25"]])
            with self.assertRaises(PredictionValidationError):
                load_prediction_file(wrong, MULTICLASS_TASK)
            invalid_sum = self.write_csv(directory, "invalid_sum.csv", [["id", "prob_cat", "prob_dog"], ["001", "0.25", "0.25"]])
            with self.assertRaises(PredictionValidationError):
                load_prediction_file(invalid_sum, MULTICLASS_TASK)

    def test_id_alignment_ignores_row_order(self):
        first = load_prediction_file(FIXTURE_DIR / "m1_oof.csv", REGRESSION_TASK)
        second = load_prediction_file(FIXTURE_DIR / "m2_oof.csv", REGRESSION_TASK)
        ids, aligned = align_predictions({"m1": first, "m2": second})
        self.assertEqual(ids, ("001", "002", "003"))
        self.assertEqual(aligned["m1"], ((1.1,), (2.0,), (3.2,)))
        self.assertEqual(aligned["m2"], ((1.0,), (2.2,), (2.8,)))

    def test_fold_validation(self):
        folds = validate_folds(FIXTURE_DIR / "folds.csv", ["001", "002", "003"], 3, ["003", "001", "002"])
        self.assertEqual(folds["001"], 0)
        with self.assertRaises(FoldValidationError):
            validate_folds(FIXTURE_DIR / "folds.csv", ["001", "002"], 3)

    def test_missing_prediction_file(self):
        config = {"prediction": {"oof_dir": "predictions/oof", "test_dir": "predictions/test"}, "task": REGRESSION_TASK}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PredictionValidationError):
                load_configured_predictions(config, "m1", directory)

    def test_oof_test_schema_consistency(self):
        config = {"prediction": {"oof_dir": "oof", "test_dir": "test"}, "task": REGRESSION_TASK}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "oof").mkdir()
            (root / "test").mkdir()
            self.write_csv(root / "oof", "m1_oof.csv", [["id", "prediction"], ["001", "1"]])
            self.write_csv(root / "test", "m1_test.csv", [["id", "score"], ["001", "1"]])
            with self.assertRaises(PredictionValidationError):
                load_configured_predictions(config, "m1", root)

    def test_metric_calculation(self):
        config = {"metric": {"name": "RMSE", "direction": "minimize"}}
        self.assertAlmostEqual(evaluate_predictions([1, 3], [2, 1], config), math.sqrt(2.5))

    def test_required_config_sections(self):
        with self.assertRaises(ConfigurationError):
            validate_config({})


if __name__ == "__main__":
    unittest.main()
