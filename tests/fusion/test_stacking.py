import unittest
from pathlib import Path

from src.fusion.stacking import nested_ridge_evaluation, predict_with_meta_model, train_meta_model
from src.utils.prediction_contract import load_prediction_file


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "m4"
CONFIG = {
    "task": {"type": "regression", "id_column": "id"},
    "metric": {"name": "RMSE", "direction": "minimize"},
    "fusion": {"ridge": {"alpha": 1.0}},
}


class Phase4StackingTests(unittest.TestCase):
    def setUp(self):
        self.predictions = {
            "m1": load_prediction_file(FIXTURE_DIR / "m1_oof.csv", CONFIG["task"]),
            "m2": load_prediction_file(FIXTURE_DIR / "m2_oof.csv", CONFIG["task"]),
        }
        self.targets = {"001": 1.0, "002": 2.0, "003": 3.0}
        self.folds = {"001": 0, "002": 1, "003": 2}

    def test_ridge_fit_and_predict(self):
        model = train_meta_model({"m1": [1.0, 2.0, 3.0], "m2": [1.0, 2.0, 3.0]}, [1.0, 2.0, 3.0], 1.0, CONFIG)
        predictions = predict_with_meta_model(model, {"m1": [1.0, 2.0], "m2": [1.0, 2.0]})
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(len(row) == 1 for row in predictions))

    def test_nested_fold_isolation(self):
        result = nested_ridge_evaluation(self.predictions, self.targets, self.folds, CONFIG)
        self.assertEqual(set(result["fold_scores"]), {"0", "1", "2"})
        self.assertEqual(len(result["predictions"]), 3)
        self.assertEqual(result["alpha"], 1.0)

    def test_alpha_validation(self):
        config = {**CONFIG, "fusion": {"ridge": {"alpha": -1.0}}}
        with self.assertRaises(ValueError):
            nested_ridge_evaluation(self.predictions, self.targets, self.folds, config)


if __name__ == "__main__":
    unittest.main()
