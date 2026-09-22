"""Run the reusable tabular baseline and write M1 prediction artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import load_train_test_data
from src.data.preprocessing import clean_train_test
from src.utils.config import load_config
from src.utils.cv_split import validate_folds


def _features(train, test, id_column, target_column):
    train_features = train.drop(columns=[column for column in (id_column, target_column) if column in train])
    test_features = test.drop(columns=[id_column], errors="ignore")
    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, dummy_na=True)
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return combined.iloc[: len(train)].to_numpy(dtype=float), combined.iloc[len(train) :].to_numpy(dtype=float)


def _estimator(task_type, seed):
    if task_type == "regression":
        return Ridge(alpha=1.0)
    if task_type in {"binary_classification", "multiclass_classification"}:
        return LogisticRegression(max_iter=1000, random_state=seed)
    raise ValueError(f"Unsupported task type: {task_type}")


def _fit_estimator(task_type, features, targets, seed):
    if task_type != "regression" and len(set(map(str, targets))) < 2:
        model = DummyClassifier(strategy="prior")
    else:
        model = _estimator(task_type, seed)
    model.fit(features, targets)
    return model


def _predict(model, features, task_type, classes, positive_class=None):
    if task_type == "regression":
        return np.asarray(model.predict(features), dtype=float)
    probabilities = model.predict_proba(features)
    model_classes = [str(value) for value in model.classes_]
    result = np.zeros((len(features), len(classes)), dtype=float)
    for index, class_name in enumerate(classes):
        if str(class_name) in model_classes:
            result[:, index] = probabilities[:, model_classes.index(str(class_name))]
    if task_type == "binary_classification":
        positive = str(positive_class) if positive_class is not None else model_classes[-1]
        return probabilities[:, model_classes.index(positive)] if positive in model_classes else probabilities[:, -1]
    return result


def _frame(ids, predictions, config):
    frame = pd.DataFrame({config["task"]["id_column"]: ids.astype(str)})
    if config["task"]["type"] == "multiclass_classification":
        for index, class_name in enumerate(config["task"]["classes"]):
            frame[f"prob_{class_name}"] = predictions[:, index]
    else:
        frame["prediction"] = predictions
    return frame


def run(config_path="config/config.yaml"):
    config_file = Path(config_path).resolve()
    root = config_file.parent.parent
    config = load_config(config_file)
    train, test = load_train_test_data(
        root / config["data"]["train_path"],
        root / config["data"]["test_path"],
        encoding=config["data"].get("encoding", "utf-8-sig"),
        strip_header_whitespace=config["data"].get("strip_header_whitespace", True),
    )
    id_column = config["task"]["id_column"]
    target_column = config["task"]["target"]
    train, test = clean_train_test(train, test, id_column=id_column)
    if target_column not in train:
        raise ValueError(f"Configured target column {target_column!r} is missing from training data")
    folds = validate_folds(
        root / config["validation"]["folds_path"],
        train[id_column],
        config["validation"]["n_folds"],
        min_valid_folds=config["validation"].get("min_valid_folds"),
    )
    train_features, test_features = _features(train, test, id_column, target_column)
    task_type = config["task"]["type"]
    classes = config["task"].get("classes", [])
    targets = train[target_column].to_numpy()
    oof = np.zeros((len(train), len(classes)), dtype=float) if task_type == "multiclass_classification" else np.zeros(len(train), dtype=float)
    fold_values = train[id_column].astype(str).map(folds).to_numpy()
    for fold in sorted(set(fold_values)):
        train_index = np.flatnonzero(fold_values != fold)
        valid_index = np.flatnonzero(fold_values == fold)
        model = _fit_estimator(task_type, train_features[train_index], targets[train_index], config.get("project", {}).get("seed", 42))
        oof[valid_index] = _predict(model, train_features[valid_index], task_type, classes, config["task"].get("positive_class"))
    model = _fit_estimator(task_type, train_features, targets, config.get("project", {}).get("seed", 42))
    test_predictions = _predict(model, test_features, task_type, classes, config["task"].get("positive_class"))
    oof_path = root / config["prediction"]["oof_dir"] / "m1_oof.csv"
    test_path = root / config["prediction"]["test_dir"] / "m1_test.csv"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    _frame(train[id_column], oof, config).to_csv(oof_path, index=False)
    _frame(test[id_column], test_predictions, config).to_csv(test_path, index=False)
    return {"oof_path": str(oof_path), "test_path": str(test_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    print(run(parser.parse_args().config))
