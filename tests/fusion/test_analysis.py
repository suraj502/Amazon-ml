import unittest
from pathlib import Path

from src.fusion.analysis import (
    disagreement_analysis,
    error_correlation,
    evaluate_individual_models,
    prediction_correlation,
    target_quantile_analysis,
)
from src.utils.prediction_contract import load_prediction_file


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "m4"
CONFIG = {
    "task": {"type": "regression", "id_column": "id"},
    "metric": {"name": "RMSE", "direction": "minimize"},
    "fusion": {"target_quantile_analysis": {"enabled": True, "quantiles": [0.5]}},
}


class Phase2AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.predictions = {
            "m1": load_prediction_file(FIXTURE_DIR / "m1_oof.csv", CONFIG["task"]),
            "m2": load_prediction_file(FIXTURE_DIR / "m2_oof.csv", CONFIG["task"]),
        }
        self.targets = {"001": 1.0, "002": 2.0, "003": 3.0}
        self.folds = {"001": 0, "002": 1, "003": 2}

    def test_individual_and_fold_scoring(self):
        report = evaluate_individual_models(self.predictions, self.targets, self.folds, CONFIG)
        self.assertEqual(set(report["models"]), {"m1", "m2"})
        self.assertEqual(report["models"]["m1"]["valid_folds"], 3)
        self.assertIn("fold_scores", report["models"]["m1"])

    def test_correlations_and_disagreement(self):
        prediction = prediction_correlation(self.predictions, CONFIG)
        errors = error_correlation(self.predictions, self.targets, CONFIG)
        disagreement = disagreement_analysis(self.predictions, CONFIG)
        self.assertIn("m1__m2", prediction)
        self.assertIn("m1__m2", errors)
        self.assertEqual(disagreement["m1__m2"]["sample_count"], 3)

    def test_target_quantile_analysis(self):
        result = target_quantile_analysis(self.predictions, self.targets, CONFIG)
        self.assertEqual(set(result["m1"]), {"Q1", "Q2"})
        self.assertEqual(sum(item["count"] for item in result["m1"].values()), 3)


if __name__ == "__main__":
    unittest.main()
