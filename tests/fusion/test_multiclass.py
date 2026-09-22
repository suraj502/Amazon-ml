import csv
import tempfile
import unittest
from pathlib import Path

from src.fusion.ensemble import simple_average
from src.fusion.final_model import generate_submission
from src.utils.metrics import evaluate_predictions
from src.utils.prediction_contract import load_prediction_file


TASK = {"type": "multiclass_classification", "id_column": "id", "classes": ["cat", "dog"]}
CONFIG = {"task": TASK, "metric": {"name": "ACCURACY", "direction": "maximize"}}


class MulticlassSupportTests(unittest.TestCase):
    def test_probability_vector_average_and_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            path_a = Path(directory) / "a.csv"
            path_b = Path(directory) / "b.csv"
            for path, rows in ((path_a, [["id", "prob_cat", "prob_dog"], ["001", "0.8", "0.2"]]), (path_b, [["id", "prob_cat", "prob_dog"], ["001", "0.6", "0.4"]])):
                with path.open("w", encoding="utf-8", newline="") as output:
                    csv.writer(output).writerows(rows)
            first = load_prediction_file(path_a, TASK)
            second = load_prediction_file(path_b, TASK)
            values = simple_average({"a": first.values, "b": second.values})
            self.assertEqual(values, [[0.7, 0.30000000000000004]])
            self.assertEqual(evaluate_predictions(["cat"], values, CONFIG), 1.0)

    def test_multiclass_submission_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.csv"
            with sample.open("w", encoding="utf-8", newline="") as output:
                csv.writer(output).writerows([["id", "prob_cat", "prob_dog"], ["001", "", ""]])
            headers, rows = generate_submission(sample, ["001"], [[0.25, 0.75]], CONFIG)
            self.assertEqual(headers, ["id", "prob_cat", "prob_dog"])
            self.assertEqual(rows[0]["prob_dog"], 0.75)


if __name__ == "__main__":
    unittest.main()
