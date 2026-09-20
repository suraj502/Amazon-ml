"""Run the M2 pipeline once the real train/test data and shared folds exist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

# Allow ``py scripts/run_m2.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.pipeline import (
    discover_text_columns,
    extract_structured_features,
    load_dataset,
    normalize_id_series,
    prepare_text,
    validate_shared_folds,
    write_report,
    write_structured_features,
)
from src.utils.config import load_config
from src.utils.metrics import evaluate_predictions


def _vectorizer(config: dict) -> TfidfVectorizer:
    settings = config.get("text", {}).get("models", {}).get("tfidf", {})
    return TfidfVectorizer(
        max_features=settings.get("max_features", 100000),
        ngram_range=tuple(settings.get("ngram_range", [1, 2])),
        min_df=settings.get("min_df", 1),
        max_df=settings.get("max_df", 1.0),
        sublinear_tf=settings.get("sublinear_tf", True),
        token_pattern=r"(?u)\b\w+\b",
    )


def _model(task_type: str):
    if task_type == "regression":
        return Ridge(alpha=1.0)
    if task_type == "binary_classification":
        return LogisticRegression(max_iter=1000, random_state=42)
    if task_type == "multiclass_classification":
        return LogisticRegression(max_iter=1000, random_state=42)
    raise ValueError(f"Unsupported task type: {task_type}")


def _fit_predict(train_text, train_target, predict_text, config):
    task_type = config["task"]["type"]
    nonempty = any(str(value).strip() for value in train_text)
    if not nonempty:
        if task_type == "regression":
            estimator = DummyRegressor(strategy="mean")
        else:
            estimator = DummyClassifier(strategy="prior")
        estimator.fit(sparse.csr_matrix((len(train_text), 1)), train_target)
        return estimator.predict(sparse.csr_matrix((len(predict_text), 1))), None

    vectorizer = _vectorizer(config)
    x_train = vectorizer.fit_transform(train_text)
    x_predict = vectorizer.transform(predict_text)
    estimator = _model(task_type)
    estimator.fit(x_train, train_target)
    if task_type == "multiclass_classification":
        return estimator.predict_proba(x_predict), vectorizer
    if task_type == "binary_classification":
        return estimator.predict_proba(x_predict)[:, 1], vectorizer
    return estimator.predict(x_predict), vectorizer


def _prediction_frame(ids, predictions, config):
    id_column = config["task"]["id_column"]
    frame = pd.DataFrame({id_column: normalize_id_series(pd.Series(ids))})
    if config["task"]["type"] == "multiclass_classification":
        for index, class_name in enumerate(config["task"]["classes"]):
            frame[f"prob_{class_name}"] = predictions[:, index]
    else:
        frame["prediction"] = predictions
    if frame.iloc[:, 1:].isna().any().any() or not np.isfinite(frame.iloc[:, 1:].to_numpy(dtype=float)).all():
        raise ValueError("M2 produced NaN or infinity predictions")
    return frame


def run(config_path: str = "config/config.yaml") -> dict:
    config = load_config(config_path)
    train = load_dataset(config["data"]["train_path"], config)
    test = load_dataset(config["data"]["test_path"], config)
    id_column = config["task"]["id_column"]
    if id_column not in train.columns or id_column not in test.columns:
        raise ValueError(f"Configured ID column {id_column!r} must exist in train and test")
    train_ids = normalize_id_series(train[id_column])
    test_ids = normalize_id_series(test[id_column])
    text_columns = discover_text_columns(train, config)
    missing_test_columns = [column for column in text_columns if column not in test.columns]
    if missing_test_columns:
        raise ValueError(f"Text columns missing from test data: {missing_test_columns}")
    folds = validate_shared_folds(config, train_ids)
    train_text = prepare_text(train, text_columns, config)
    test_text = prepare_text(test, text_columns, config)

    structured = extract_structured_features(train_ids, train_text, config)
    structured_path = config["text"]["structured_extraction"]["output_path"]
    write_structured_features(
        structured,
        structured_path,
        overwrite=config.get("runtime", {}).get("overwrite_features", False),
    )

    target_column = config["task"]["target"]
    if target_column not in train.columns:
        raise ValueError(f"Configured target column {target_column!r} is missing from train data")
    target = train[target_column]
    oof = np.empty((len(train), len(config["task"].get("classes", [])) or 1), dtype=float)
    if config["task"]["type"] != "multiclass_classification":
        oof = oof[:, 0]
    fold_series = train_ids.map(folds).to_numpy()
    for fold in sorted(set(fold_series)):
        train_index = np.flatnonzero(fold_series != fold)
        valid_index = np.flatnonzero(fold_series == fold)
        predictions, _ = _fit_predict(
            train_text.iloc[train_index].tolist(),
            target.iloc[train_index].to_numpy(),
            train_text.iloc[valid_index].tolist(),
            config,
        )
        oof[valid_index] = predictions
    test_predictions, _ = _fit_predict(
        train_text.tolist(), target.to_numpy(), test_text.tolist(), config
    )
    oof_frame = _prediction_frame(train_ids, oof, config)
    test_frame = _prediction_frame(test_ids, test_predictions, config)
    oof_path = Path(config["prediction"]["oof_dir"]) / "m2_oof.csv"
    test_path = Path(config["prediction"]["test_dir"]) / "m2_test.csv"
    if not config.get("runtime", {}).get("overwrite_predictions", False) and (oof_path.exists() or test_path.exists()):
        raise FileExistsError("Refusing to overwrite existing M2 prediction files")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    oof_frame.to_csv(oof_path, index=False)
    test_frame.to_csv(test_path, index=False)
    metric_predictions = oof.tolist()
    metric_score = evaluate_predictions(target.tolist(), metric_predictions, config)
    metrics = {
        "model": "m2",
        "metric": config["metric"]["name"],
        "direction": config["metric"]["direction"],
        "oof_score": metric_score,
        "valid_folds": len(set(fold_series.tolist())),
    }
    reports_dir = Path(config.get("paths", {}).get("reports", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "m2_metrics.json").write_text(
        pd.Series(metrics).to_json(indent=2), encoding="utf-8"
    )
    (reports_dir / "m2_metrics.md").write_text(
        "\n".join(["# M2 metrics", ""] + [f"- **{key}:** {value}" for key, value in metrics.items()]) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{
        "experiment": "tfidf_baseline",
        "change": "TF-IDF with fold-fitted estimator",
        "why": "Cheap leakage-safe baseline",
        "metric": config["metric"]["name"],
        "score": metric_score,
        "compute_cost": "low",
        "decision": "prepared; reassess after real validation",
    }]).to_csv(reports_dir / "m2_experiments.csv", index=False)
    report = {
        "text_columns": text_columns,
        "train_rows": len(train),
        "test_rows": len(test),
        "missing_train_text_rows": int(train_text.eq("").sum()),
        "missing_test_text_rows": int(test_text.eq("").sum()),
        "folds_used": sorted(set(fold_series.tolist())),
        "model": "TF-IDF + configured task estimator",
        "structured_output": str(structured_path),
        "oof_score": metric_score,
    }
    write_report(
        report,
        reports_dir / "m2_text_report.json",
        reports_dir / "m2_text_report.md",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)
