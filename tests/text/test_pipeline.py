import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.text.pipeline import extract_structured_features, load_dataset


class M2PipelineTests(unittest.TestCase):
    def test_dataset_loader_preserves_leading_zero_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            pd.DataFrame({"id": ["001"], "target": [1.0]}).to_csv(path, index=False)
            config = {"data": {"encoding": "utf-8-sig"}, "task": {"id_column": "id"}}

            loaded = load_dataset(path, config)

            self.assertEqual(loaded["id"].tolist(), ["001"])

    def test_dimensions_extraction_supports_multiple_numbers(self):
        config = {
            "task": {"id_column": "id"},
            "text": {"structured_extraction": {"fields": {"dimensions": {"enabled": True}}}},
        }

        features = extract_structured_features(
            pd.Series(["001"]), pd.Series(["box 10x20x30 cm"]), config
        )

        self.assertEqual(features.loc[0, "dimensions"], "10x20x30 cm")


if __name__ == "__main__":
    unittest.main()