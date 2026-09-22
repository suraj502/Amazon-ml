"""Simple and constrained prediction blending for M4."""

import itertools
import math
import warnings

from src.utils.metrics import evaluate_predictions
from src.utils.prediction_contract import align_predictions as align_prediction_tables


def load_predictions(paths, task=None):
    """Load explicitly named prediction files; never auto-discover files."""
    if task is None:
        raise ValueError("task configuration is required to load predictions")
    from src.utils.prediction_contract import load_prediction_file

    return {name: load_prediction_file(path, task) for name, path in paths.items()}


def align_predictions(predictions, expected_ids=None):
    """Delegate ID alignment to the Phase 1 contract validator."""
    return align_prediction_tables(predictions, expected_ids=expected_ids)


def simple_average(predictions):
    """Return an equal-weight blend, including the one-model passthrough."""
    vectors = _validate_vectors(predictions)
    model_count = len(vectors)
    row_count = len(next(iter(vectors.values())))
    return [_average_rows([vectors[model][row] for model in vectors]) for row in range(row_count)]


def weighted_average(predictions, weights):
    """Return a weighted blend after validating nonnegative unit-sum weights."""
    vectors = _validate_vectors(predictions)
    models = list(vectors)
    if len(weights) != len(models) or any(weight < 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("Weights must be finite, nonnegative, and match the model count")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Weights must sum to one within 1e-9")
    return [_weighted_rows([vectors[model][row] for model in models], weights) for row in range(len(next(iter(vectors.values()))))]


def optimize_weights(predictions, targets, config, blend_space=None):
    """Optimize simplex weights deterministically using OOF predictions only."""
    vectors = _validate_vectors(predictions)
    if config["task"]["type"] == "multiclass_classification":
        targets = list(targets)
    else:
        targets = [float(value) for value in targets]
    if len(targets) != len(next(iter(vectors.values()))):
        raise ValueError("Targets and prediction rows must have the same length")
    space = blend_space or config.get("fusion", {}).get("prediction_space", {}).get("default", "raw")
    transformed = {model: transform_space(values, space, config["task"], inverse=False) for model, values in vectors.items()}
    if any(isinstance(row, (list, tuple)) for row in next(iter(transformed.values()))):
        if space != "raw":
            raise ValueError("Non-raw blend spaces are not supported for multiclass probability vectors")
        classes = config["task"].get("classes", [])
        transformed_targets = [[1.0 if str(target) == str(class_name) else 0.0 for class_name in classes] for target in targets]
    else:
        transformed_targets = transform_space(targets, space, config["task"], inverse=False, target=True)
    models = list(transformed)
    weights = [1.0 / len(models)] * len(models)
    try:
        scale = max(max(abs(item) for row in transformed[model] for item in _normalize_row(row)) for model in models) or 1.0
        learning_rate = 0.25 / scale
        for _ in range(1000):
            blended = weighted_average(transformed, weights)
            gradient = []
            for model in models:
                gradient.append(2.0 * sum(_row_dot_error(blended[row], transformed_targets[row], transformed[model][row]) for row in range(len(targets))) / len(targets))
            updated = _project_simplex([weight - learning_rate * value for weight, value in zip(weights, gradient)])
            if max(abs(left - right) for left, right in zip(updated, weights)) < 1e-10:
                weights = updated
                break
            weights = updated
        result = weighted_average(transformed, weights)
        result_values = [item for row in result for item in _normalize_row(row)]
        if not all(math.isfinite(value) for value in weights + result_values):
            raise ArithmeticError("optimizer produced non-finite values")
        return {"weights": dict(zip(models, weights)), "status": "success", "blend_space": space}
    except (ArithmeticError, ValueError, ZeroDivisionError) as exc:
        warnings.warn(f"Weighted optimizer failed; using equal weights: {exc}", RuntimeWarning)
        equal = 1.0 / len(models)
        return {"weights": dict.fromkeys(models, equal), "status": "fallback_equal_weight", "blend_space": space}


def fit_weighted_blend(predictions, targets, config, blend_space=None):
    """Fit weights on OOF rows and score the resulting original-scale blend."""
    vectors = _validate_vectors(predictions)
    result = optimize_weights(vectors, targets, config, blend_space)
    weights = [result["weights"][model] for model in vectors]
    space = result["blend_space"]
    transformed = {model: transform_space(values, space, config["task"], inverse=False) for model, values in vectors.items()}
    blended_space = weighted_average(transformed, weights)
    blended = transform_space(blended_space, space, config["task"], inverse=True)
    result["predictions"] = blended
    result["score"] = evaluate_predictions(targets, blended, config)
    return result


def evaluate_ensemble(predictions, targets, config=None, weights=None):
    """Score an equal or explicitly weighted blend with the shared metric."""
    blended = simple_average(predictions) if weights is None else weighted_average(predictions, weights)
    return evaluate_predictions(targets, blended, config=config)


def evaluate_ablations(predictions, targets, config, method="simple_average"):
    """Evaluate every non-empty configured-model subset deterministically."""
    models = list(predictions)
    results = {}
    for size in range(1, len(models) + 1):
        for subset in itertools.combinations(models, size):
            subset_predictions = {model: predictions[model] for model in subset}
            if method == "weighted_blend" and len(subset) > 1:
                result = fit_weighted_blend(subset_predictions, targets, config)
                results["+".join(subset)] = result
            else:
                values = simple_average(subset_predictions)
                results["+".join(subset)] = {"predictions": values, "score": evaluate_predictions(targets, values, config)}
    return results


def transform_space(values, space, task, inverse=False, target=False):
    """Transform supported blend spaces without guessing target scale."""
    if any(isinstance(value, (list, tuple)) for value in values):
        if space != "raw":
            raise ValueError("Non-raw blend spaces are not supported for multiclass probability vectors")
        return [list(value) for value in values]
    values = [float(value) for value in values]
    if space == "raw":
        return values
    if space == "log1p":
        if task["type"] != "regression":
            raise ValueError("log1p blend space is only valid for regression")
        if inverse:
            return [math.expm1(value) for value in values]
        if any(value < 0 for value in values):
            raise ValueError("log1p blend space requires nonnegative values")
        return [math.log1p(value) for value in values]
    if space == "logit":
        if task["type"] != "binary_classification":
            raise ValueError("logit blend space is only valid for binary classification")
        if inverse:
            return [1.0 / (1.0 + math.exp(-max(min(value, 700.0), -700.0))) for value in values]
        if any(value <= 0 or value >= 1 for value in values):
            raise ValueError("logit blend space requires probabilities strictly between zero and one")
        return [math.log(value / (1.0 - value)) for value in values]
    if space == "rank":
        if inverse or task["type"] != "regression":
            raise ValueError("rank blend space is only supported as an explicit regression input transform")
        ordering = sorted(range(len(values)), key=values.__getitem__)
        ranks = [0.0] * len(values)
        for rank, index in enumerate(ordering):
            ranks[index] = rank / max(len(values) - 1, 1)
        return ranks
    raise ValueError(f"Unsupported blend space: {space!r}")


def _validate_vectors(predictions):
    if not predictions:
        raise ValueError("At least one model is required")
    vectors = {model: [_normalize_row(value) for value in values] for model, values in predictions.items()}
    row_count = len(next(iter(vectors.values())))
    if any(len(values) != row_count for values in vectors.values()):
        raise ValueError("All prediction vectors must have the same row count")
    if any(not math.isfinite(value) for values in vectors.values() for row in values for value in row):
        raise ValueError("Predictions must be finite")
    return vectors


def _normalize_row(value):
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _average_rows(rows):
    width = len(rows[0])
    result = [sum(row[index] for row in rows) / len(rows) for index in range(width)]
    return result[0] if width == 1 else result


def _weighted_rows(rows, weights):
    width = len(rows[0])
    result = [sum(weight * row[index] for weight, row in zip(weights, rows)) for index in range(width)]
    return result[0] if width == 1 else result


def _row_dot_error(prediction, target, feature):
    prediction_row = _normalize_row(prediction)
    target_row = _normalize_row(target)
    feature_row = _normalize_row(feature)
    return sum((left - right) * value for left, right, value in zip(prediction_row, target_row, feature_row))


def _project_simplex(values):
    """Project a vector onto nonnegative weights summing to one."""
    sorted_values = sorted(values, reverse=True)
    cumulative = 0.0
    threshold = 0.0
    for index, value in enumerate(sorted_values, 1):
        cumulative += value
        candidate = (cumulative - 1.0) / index
        if value - candidate > 0:
            threshold = candidate
    return [max(value - threshold, 0.0) for value in values]
