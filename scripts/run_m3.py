"""Run the reusable image embedding baseline and write M3 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.image.image_embedding_pipeline import ImageEmbeddingPipeline
from src.image.image_predictor import ImagePredictor
from src.utils.config import load_config
from src.utils.cv_split import validate_folds


def run(config_path="config/config.yaml"):
    config_file = Path(config_path).resolve()
    root = config_file.parent.parent
    config = load_config(config_file)
    image_config = config.get("image", {})
    reference_column = image_config.get("path_column") or image_config.get("url_column")
    if not reference_column:
        raise ValueError("Configure image.path_column or image.url_column before running M3")
    train_path = root / config["data"]["train_path"]
    test_path = root / config["data"]["test_path"]
    train = pd.read_csv(train_path, encoding=config["data"].get("encoding", "utf-8-sig"), dtype={config["task"]["id_column"]: str})
    test = pd.read_csv(test_path, encoding=config["data"].get("encoding", "utf-8-sig"), dtype={config["task"]["id_column"]: str})
    id_column = config["task"]["id_column"]
    target_column = config["task"]["target"]
    if reference_column not in train or reference_column not in test:
        raise ValueError(f"Configured image reference column {reference_column!r} must exist in train and test")
    folds = validate_folds(
        root / config["validation"]["folds_path"],
        train[id_column],
        config["validation"]["n_folds"],
        min_valid_folds=config["validation"].get("min_valid_folds"),
    )
    pipeline = ImageEmbeddingPipeline(cache_dir=root / image_config.get("embeddings_dir", "features/image_embeddings"))
    train_embeddings, train_statuses = pipeline.process(train[reference_column].tolist(), image_config.get("batch_size", 32))
    test_embeddings, test_statuses = pipeline.process(test[reference_column].tolist(), image_config.get("batch_size", 32))
    valid_train = np.array([status["status"] not in {"missing", "invalid"} for status in train_statuses])
    valid_test = np.array([status["status"] not in {"missing", "invalid"} for status in test_statuses])
    targets = train[target_column].to_numpy(dtype=float)
    fold_values = train[id_column].astype(str).map(folds).to_numpy()
    oof = np.zeros(len(train), dtype=float)
    for fold in sorted(set(fold_values)):
        train_index = np.flatnonzero((fold_values != fold) & valid_train)
        valid_index = np.flatnonzero(fold_values == fold)
        predictor = ImagePredictor(alpha=1.0)
        if len(train_index):
            predictor.fit(train_embeddings[train_index], targets[train_index], valid_mask=np.ones(len(train_index), dtype=bool))
            oof[valid_index] = predictor.predict(train_embeddings[valid_index], valid_mask=valid_train[valid_index])
        else:
            oof[valid_index] = np.mean(targets[valid_train]) if valid_train.any() else 0.0
    predictor = ImagePredictor(alpha=1.0)
    predictor.fit(train_embeddings, targets, valid_mask=valid_train)
    test_predictions = predictor.predict(test_embeddings, valid_mask=valid_test)
    oof_path = root / config["prediction"]["oof_dir"] / "m3_oof.csv"
    test_path = root / config["prediction"]["test_dir"] / "m3_test.csv"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({id_column: train[id_column].astype(str), "prediction": oof}).to_csv(oof_path, index=False)
    pd.DataFrame({id_column: test[id_column].astype(str), "prediction": test_predictions}).to_csv(test_path, index=False)
    return {"oof_path": str(oof_path), "test_path": str(test_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    print(run(parser.parse_args().config))
