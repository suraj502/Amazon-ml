import csv
import tempfile
import unittest
from pathlib import Path

from src.fusion.ensemble import fit_weighted_blend
from src.fusion.final_model import compare_candidates
from src.utils.cv_split import FoldValidationError, validate_folds


class PipelineBlockerTests(unittest.TestCase):
    def test_multiclass_weighted_blend_accepts_string_targets(self):
        config = {
            "task": {"type": "multiclass_classification", "classes": ["cat", "dog"]},
            "metric": {"name": "logloss", "direction": "minimize"},
            "fusion": {
                "prediction_space": {"default": "raw"},
                "weighted_blend": {
                    "nonnegative": True,
                    "sum_to_one": True,
                    "tolerance": 1e-9,
                    "fallback": "equal_weight",
                },
            },
        }
        result = fit_weighted_blend(
            {"m1": [[0.9, 0.1], [0.1, 0.9]], "m2": [[0.8, 0.2], [0.2, 0.8]]},
            ["cat", "dog"],
            config,
        )
        self.assertEqual(len(result["predictions"]), 2)
        self.assertAlmostEqual(sum(result["predictions"][0]), 1.0)

    def test_disabled_fusion_candidates_are_not_evaluated(self):
        config = {
            "task": {"type": "regression"},
            "metric": {"name": "RMSE", "direction": "minimize"},
            "fusion": {
                "candidate_methods": {
                    "simple_average": {"enabled": False},
                    "weighted_blend": {"enabled": False},
                    "ridge": {"enabled": False},
                },
                "ridge": {"enabled": False},
            },
        }
        predictions = {
            "m1": _table([("a", 1.0), ("b", 2.0)]),
            "m2": _table([("a", 1.5), ("b", 2.5)]),
        }
        result = compare_candidates(predictions, {"a": 1.0, "b": 2.0}, {"a": 0, "b": 1}, config)
        self.assertEqual(list(result["candidates"]), ["m1", "m2"])

    def test_minimum_valid_folds_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(["id", "fold"])
                writer.writerow(["a", 0])
                writer.writerow(["b", 0])
            with self.assertRaises(FoldValidationError):
                validate_folds(path, ["a", "b"], 2, min_valid_folds=2)


def _table(rows):
    from src.utils.prediction_contract import PredictionTable

    return PredictionTable(
        ids=tuple(identifier for identifier, _ in rows),
        columns=("prediction",),
        values=tuple((value,) for _, value in rows),
        path=Path("test.csv"),
    )


if __name__ == "__main__":
    unittest.main()
