import math
import unittest
import warnings
from unittest import mock

import src.fusion.ensemble as ensemble
from src.fusion.ensemble import (
    evaluate_ablations,
    fit_weighted_blend,
    optimize_weights,
    simple_average,
    transform_space,
    weighted_average,
)


CONFIG = {
    "task": {"type": "regression", "id_column": "id"},
    "metric": {"name": "RMSE", "direction": "minimize"},
    "fusion": {"prediction_space": {"default": "raw"}},
}


class Phase3EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.predictions = {"m1": [1.0, 2.0, 3.0], "m2": [1.4, 1.8, 2.6]}
        self.targets = [1.0, 2.0, 3.0]

    def test_simple_average_and_weight_constraints(self):
        self.assertEqual(simple_average({"m1": [1.0], "m2": [3.0]}), [2.0])
        self.assertEqual(weighted_average(self.predictions, [1.0, 0.0]), self.predictions["m1"])
        with self.assertRaises(ValueError):
            weighted_average(self.predictions, [0.5, 0.6])

    def test_weighted_optimization_and_score(self):
        result = fit_weighted_blend(self.predictions, self.targets, CONFIG)
        self.assertEqual(result["status"], "success")
        self.assertAlmostEqual(sum(result["weights"].values()), 1.0)
        self.assertTrue(math.isfinite(result["score"]))

    def test_one_model_and_zero_model(self):
        self.assertEqual(simple_average({"m1": [1.0, 2.0]}), [1.0, 2.0])
        with self.assertRaises(ValueError):
            simple_average({})

    def test_optimizer_fallback(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch.object(ensemble, "_project_simplex", side_effect=ArithmeticError("test failure")):
                result = optimize_weights({"m1": [1.0], "m2": [2.0]}, [1.0], CONFIG)
        self.assertEqual(result["status"], "fallback_equal_weight")
        self.assertTrue(any("optimizer failed" in str(item.message) for item in caught))

    def test_blend_spaces_and_ablations(self):
        log_values = transform_space([0.0, 1.0], "log1p", CONFIG["task"])
        self.assertEqual(transform_space(log_values, "log1p", CONFIG["task"], inverse=True), [0.0, 1.0])
        result = evaluate_ablations(self.predictions, self.targets, CONFIG)
        self.assertEqual(set(result), {"m1", "m2", "m1+m2"})


if __name__ == "__main__":
    unittest.main()
