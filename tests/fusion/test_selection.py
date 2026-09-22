import csv
import tempfile
import unittest
from pathlib import Path

from src.fusion.final_model import (
    compare_candidates,
    distribution_shift,
    fit_selected_candidate,
    generate_final_predictions,
    generate_submission,
    select_best_candidate,
)
from src.utils.prediction_contract import load_prediction_file


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "m4"
CONFIG = {
    "task": {"type": "regression", "id_column": "id"},
    "metric": {"name": "RMSE", "direction": "minimize"},
    "fusion": {
        "prediction_space": {"default": "raw"},
        "ridge": {"alpha": 1.0},
        "selection": {"one_se_rule": False},
        "target_quantile_analysis": {"enabled": True, "quantiles": [0.5]},
    },
}


class Phase5And6Tests(unittest.TestCase):
    def setUp(self):
        self.oof = {
            "m1": load_prediction_file(FIXTURE_DIR / "m1_oof.csv", CONFIG["task"]),
            "m2": load_prediction_file(FIXTURE_DIR / "m2_oof.csv", CONFIG["task"]),
        }
        self.test = {"m1": load_prediction_file(FIXTURE_DIR / "m1_test.csv", CONFIG["task"])}
        self.targets = {"002": 2.0, "001": 1.0, "003": 3.0}
        self.folds = {"001": 0, "002": 1, "003": 2}

    def test_candidate_comparison_and_selection(self):
        report = compare_candidates(self.oof, self.targets, self.folds, CONFIG)
        selection = select_best_candidate(report["candidates"], CONFIG)
        self.assertIn("selected", selection)
        self.assertIsNotNone(selection["best_single"])

    def test_final_single_model_fusion_and_shift(self):
        candidate = {"method": "single", "models": ["m1"]}
        fitted = fit_selected_candidate(candidate, self.oof, self.targets, CONFIG)
        ids, values = generate_final_predictions(fitted, self.test, CONFIG)
        self.assertEqual(ids, ("T001", "T002"))
        self.assertEqual(values, [1.2, 2.1])
        shift = distribution_shift(self.oof["m1"], self.test["m1"])
        self.assertIn("warning", shift)

    def test_submission_preserves_sample_order(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.csv"
            with sample.open("w", encoding="utf-8", newline="") as output:
                csv.writer(output).writerows([["id", "target"], ["T002", ""], ["T001", ""]])
            headers, rows = generate_submission(sample, ["T001", "T002"], [1.2, 2.1], CONFIG)
            self.assertEqual(headers, ["id", "target"])
            self.assertEqual([row["id"] for row in rows], ["T002", "T001"])
            self.assertEqual([row["target"] for row in rows], [2.1, 1.2])

    def test_submission_id_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.csv"
            with sample.open("w", encoding="utf-8", newline="") as output:
                csv.writer(output).writerows([["id", "target"], ["T001", ""]])
            with self.assertRaises(ValueError):
                generate_submission(sample, ["T002"], [2.0], CONFIG)


if __name__ == "__main__":
    unittest.main()
