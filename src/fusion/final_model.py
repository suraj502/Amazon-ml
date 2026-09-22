"""Candidate selection, final test fusion, and submission validation."""

import csv
import json
import math
import statistics
import warnings
from collections import OrderedDict
from pathlib import Path

from src.fusion.analysis import evaluate_individual_models
from src.fusion.ensemble import fit_weighted_blend, simple_average, weighted_average
from src.fusion.stacking import nested_ridge_evaluation, predict_with_meta_model, train_meta_model
from src.utils.metrics import evaluate_predictions
from src.utils.prediction_contract import PredictionValidationError, align_predictions
from src.utils.cv_split import validate_folds


class SelectionError(ValueError):
    """Raised when no valid final candidate exists."""


def compare_candidates(predictions, targets, folds, config):
    """Evaluate individual, simple, weighted, Ridge, and ablation candidates."""
    ids, aligned = align_predictions(predictions, expected_ids=targets.keys())
    vectors = {model: [list(row) if config["task"]["type"] == "multiclass_classification" else row[0] for row in aligned[model]] for model in predictions}
    candidates = OrderedDict()
    individual_report = evaluate_individual_models(predictions, targets, folds, config)
    for model, report in individual_report["models"].items():
        candidates[model] = _candidate(
            model,
            "single",
            report["overall_score"],
            report["fold_scores"],
            report["fold_std"],
            [model],
        )

    methods = config.get("fusion", {}).get("candidate_methods", {})
    if methods.get("simple_average", {}).get("enabled", True):
        simple_values = simple_average(vectors)
        candidates["simple_average"] = _candidate(
            "simple_average",
            "simple_average",
            evaluate_predictions([targets[identifier] for identifier in ids], simple_values, config),
            _fold_scores(simple_values, ids, targets, folds, config),
            None,
            list(predictions),
        )
    if methods.get("weighted_blend", {}).get("enabled", True):
        weighted = fit_weighted_blend(vectors, list(targets[identifier] for identifier in ids), config)
        candidates["weighted_blend"] = _candidate(
            "weighted_blend",
            "weighted_blend",
            weighted["score"],
            _fold_scores(weighted["predictions"], ids, targets, folds, config),
            None,
            list(predictions),
            weights=weighted["weights"],
            status=weighted["status"],
        )
    if methods.get("ridge", {}).get("enabled", True) and config.get("fusion", {}).get("ridge", {}).get("enabled", True):
        ridge = nested_ridge_evaluation(predictions, targets, folds, config)
        candidates["ridge"] = _candidate(
            "ridge",
            "ridge",
            ridge["overall_score"],
            ridge["fold_scores"],
            ridge["fold_std"],
            list(predictions),
            alpha=ridge["alpha"],
        )
    return {"individual_analysis": individual_report, "candidates": candidates}


def run_m4(config, targets, root=None, sample_submission_path=None, write_submission=False):
    """Run configured M4 over supplied targets without creating artifacts by default."""
    base = Path(root) if root is not None else Path.cwd()
    models = config["fusion"]["models"]
    from src.utils.prediction_contract import load_configured_predictions

    oof_predictions = {}
    test_predictions = {}
    for model in models:
        oof, test = load_configured_predictions(config, model, base)
        oof_predictions[model] = oof
        test_predictions[model] = test
    fold_path = base / config["validation"]["folds_path"]
    folds = validate_folds(fold_path, targets.keys(), config["validation"]["n_folds"], next(iter(oof_predictions.values())).ids)
    comparison = compare_candidates(oof_predictions, targets, folds, config)
    selection = select_best_candidate(comparison["candidates"], config)
    fitted = fit_selected_candidate(selection["selected"], oof_predictions, targets, config)
    test_ids, test_values = generate_final_predictions(fitted, test_predictions, config)
    result = {"comparison": comparison, "selection": selection, "fitted": fitted, "test_ids": test_ids, "test_predictions": test_values}
    if sample_submission_path is not None:
        output_path = configured_submission_path(config, base) if write_submission else None
        result["submission"] = generate_submission(sample_submission_path, test_ids, test_values, config, output_path)
    return result


def select_best_candidate(candidates, config):
    """Select deterministically, respecting metric direction and tie breakers."""
    valid = [candidate for candidate in candidates.values() if _finite(candidate.get("score"))]
    if not valid:
        raise SelectionError("No candidate has a finite validation score")
    direction = config["metric"]["direction"]
    reverse = direction == "maximize"
    one_se = config.get("fusion", {}).get("selection", {}).get("one_se_rule", False)
    ordered = sorted(valid, key=lambda item: (_score_key(item, direction), item["fold_std"] if item["fold_std"] is not None else float("inf"), len(item["models"]), item["name"]))
    best = ordered[0]
    if one_se and len(valid) > 1:
        best_std = best["fold_std"] or 0.0
        threshold = best["score"] + best_std if direction == "minimize" else best["score"] - best_std
        eligible = [candidate for candidate in valid if candidate["score"] <= threshold] if direction == "minimize" else [candidate for candidate in valid if candidate["score"] >= threshold]
        best = sorted(eligible, key=lambda item: (len(item["models"]), item["fold_std"] if item["fold_std"] is not None else float("inf"), item["name"]))[0]
    best_single = min((candidate for candidate in valid if candidate["method"] == "single"), key=lambda item: _score_key(item, direction), default=None)
    return {
        "selected": best,
        "best_single": best_single,
        "reason": "best finite configured candidate by metric direction and deterministic tie-breaks",
        "metric_direction": direction,
    }


def fit_selected_candidate(candidate, oof_predictions, targets, config):
    """Fit a selected candidate on all OOF rows for final test prediction."""
    ids, aligned = align_predictions(oof_predictions, expected_ids=targets.keys())
    vectors = {model: [list(row) if config["task"]["type"] == "multiclass_classification" else row[0] for row in aligned[model]] for model in oof_predictions}
    if candidate["method"] == "single":
        return {"method": "single", "model": candidate["models"][0]}
    if candidate["method"] == "simple_average":
        return {"method": "simple_average", "models": list(vectors)}
    if candidate["method"] == "weighted_blend":
        fitted = fit_weighted_blend(vectors, [targets[identifier] for identifier in ids], config)
        return {"method": "weighted_blend", "models": list(vectors), "weights": fitted["weights"], "blend_space": fitted["blend_space"]}
    if candidate["method"] == "ridge":
        model = train_meta_model(oof_predictions={name: aligned[name] for name in oof_predictions}, targets=[targets[identifier] for identifier in ids], alpha=candidate.get("alpha", 1.0), config=config)
        return {"method": "ridge", "models": list(vectors), "model": model}
    raise SelectionError(f"Unsupported final candidate method: {candidate['method']!r}")


def generate_final_predictions(fitted, test_predictions, config):
    """Apply a fitted candidate to validated test prediction tables."""
    ids, aligned = align_predictions(test_predictions)
    vectors = {model: [list(row) if config["task"]["type"] == "multiclass_classification" else row[0] for row in aligned[model]] for model in test_predictions}
    method = fitted["method"]
    if method == "single":
        values = vectors[fitted["model"]]
    elif method == "simple_average":
        values = simple_average({model: vectors[model] for model in fitted["models"]})
    elif method == "weighted_blend":
        values = weighted_average({model: vectors[model] for model in fitted["models"]}, [fitted["weights"][model] for model in fitted["models"]])
    elif method == "ridge":
        values = predict_with_meta_model(fitted["model"], {model: table.values for model, table in test_predictions.items()})
    else:
        raise SelectionError(f"Unsupported final prediction method: {method!r}")
    return ids, values


def distribution_shift(oof_table, test_table):
    """Return deterministic OOF/test distribution diagnostics and warnings."""
    oof = [value for row in oof_table.values for value in row]
    test = [value for row in test_table.values for value in row]
    if not oof or not test:
        return {"warning": False, "reason": "empty distribution"}
    result = {
        "oof": _distribution(oof),
        "test": _distribution(test),
        "mean_delta": statistics.fmean(test) - statistics.fmean(oof),
        "std_delta": _std(test) - _std(oof),
    }
    result["warning"] = abs(result["mean_delta"]) > max(_std(oof), 1e-12) * 3 or abs(result["std_delta"]) > max(_std(oof), 1e-12) * 3
    if result["warning"]:
        warnings.warn("OOF/test prediction distribution shift detected", RuntimeWarning)
    return result


def generate_submission(sample_path, prediction_ids, values, config, output_path=None):
    """Validate and optionally write a submission in sample row order."""
    sample_file = Path(sample_path)
    if not sample_file.is_file():
        raise PredictionValidationError(f"Sample submission does not exist: {sample_file}")
    with sample_file.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        raw_headers = reader.fieldnames or []
        headers = [str(header).strip() for header in raw_headers]
        id_column = config["task"]["id_column"]
        if id_column not in headers:
            raise PredictionValidationError(f"Sample submission is missing ID column {id_column!r}")
        output_columns = [header for header in headers if header != id_column]
        task_type = config["task"]["type"]
        expected_output_count = len(config["task"].get("classes", [])) if task_type == "multiclass_classification" else 1
        if len(output_columns) != expected_output_count:
            raise PredictionValidationError("Sample submission columns do not match the configured task contract")
        rows = list(reader)
    by_id = {}
    for identifier, value in zip(prediction_ids, values):
        normalized = str(identifier).strip()
        if normalized in by_id:
            raise PredictionValidationError(f"Duplicate final prediction ID: {normalized!r}")
        if isinstance(value, (list, tuple)):
            numeric_value = [float(item) for item in value]
            if any(not math.isfinite(item) for item in numeric_value):
                raise PredictionValidationError(f"Invalid final prediction for ID {normalized!r}")
            by_id[normalized] = numeric_value
        else:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise PredictionValidationError(f"Invalid final prediction for ID {normalized!r}")
            by_id[normalized] = numeric_value
    sample_ids = [str(row[id_column]).strip() for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise PredictionValidationError("Sample submission contains duplicate IDs")
    if set(sample_ids) != set(by_id):
        raise PredictionValidationError("Final prediction IDs do not match sample submission IDs")
    output_rows = []
    for row, identifier in zip(rows, sample_ids):
        value = by_id[identifier]
        if task_type == "binary_classification" and not 0 <= value <= 1:
            raise PredictionValidationError("Classification submission probabilities must be in [0, 1]")
        if task_type == "multiclass_classification":
            if not isinstance(value, list) or len(value) != expected_output_count or any(not 0 <= item <= 1 for item in value) or not math.isclose(sum(value), 1.0, abs_tol=1e-6):
                raise PredictionValidationError("Multiclass submission probabilities are invalid")
        output = {header: row[raw_headers[index]] for index, header in enumerate(headers) if header != id_column}
        output[id_column] = row[raw_headers[headers.index(id_column)]]
        if task_type == "multiclass_classification":
            for column, item in zip(output_columns, value):
                output[column] = item
        else:
            output[output_columns[0]] = value
        output_rows.append(output)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(output_rows)
    return headers, output_rows


def configured_submission_path(config, root=None):
    """Resolve the configured final filename under the canonical submissions directory."""
    filename = config.get("fusion", {}).get("final_submission_filename")
    if not isinstance(filename, str) or not filename.strip():
        raise SelectionError("fusion.final_submission_filename must be configured before writing a submission")
    if Path(filename).name != filename:
        raise SelectionError("fusion.final_submission_filename must be a filename, not a path")
    base = Path(root) if root is not None else Path.cwd()
    return base / config.get("paths", {}).get("submissions", "submissions") / filename


def write_json_report(path, report):
    """Write a deterministic JSON report when explicitly requested by a caller."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True, allow_nan=False)


def train_final_model(features, targets):
    """Deprecated compatibility shim; M4 does not retrain base models."""
    raise NotImplementedError("M4 consumes base predictions; it does not train base models")


def _candidate(name, method, score, fold_scores, fold_std, models, weights=None, alpha=None, status=None):
    finite_scores = [float(value) for value in fold_scores.values() if _finite(value)]
    return {
        "name": name,
        "method": method,
        "models": models,
        "score": float(score),
        "fold_scores": fold_scores,
        "fold_mean": statistics.fmean(finite_scores) if finite_scores else None,
        "fold_std": fold_std if fold_std is not None else statistics.pstdev(finite_scores) if len(finite_scores) > 1 else 0.0,
        "weights": weights,
        "alpha": alpha,
        "status": status or "success",
    }


def _fold_scores(values, ids, targets, folds, config):
    result = OrderedDict()
    for fold in sorted(set(folds[identifier] for identifier in ids)):
        selected = [index for index, identifier in enumerate(ids) if folds[identifier] == fold]
        result[str(fold)] = evaluate_predictions([targets[ids[index]] for index in selected], [values[index] for index in selected], config)
    return result


def _score_key(candidate, direction):
    score = candidate["score"]
    return -score if direction == "maximize" else score


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _distribution(values):
    return {"count": len(values), "min": min(values), "max": max(values), "mean": statistics.fmean(values), "std": _std(values), "constant": len(set(values)) == 1}


def _std(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0
