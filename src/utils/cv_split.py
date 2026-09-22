"""Validation helpers for the shared, precomputed fold assignment."""

import csv
import warnings
from pathlib import Path


class FoldValidationError(ValueError):
    """Raised when the shared fold file violates its contract."""


def validate_folds(
    path,
    expected_ids,
    n_folds,
    oof_ids=None,
    warn_small_fold_size=None,
    min_valid_folds=None,
):
    """Validate and return ``id -> fold`` without creating any folds."""
    fold_path = Path(path)
    if not fold_path.is_file():
        raise FoldValidationError(f"Shared folds file does not exist: {fold_path}")
    expected = {_normalize_id(value) for value in expected_ids}
    folds = {}
    with fold_path.open("r", encoding="utf-8-sig", newline="") as fold_file:
        reader = csv.DictReader(fold_file)
        headers = [_header.strip() for _header in (reader.fieldnames or [])]
        if headers != ["id", "fold"]:
            raise FoldValidationError("Shared folds must have exactly the columns: id,fold")
        for row in reader:
            normalized_row = {_header(key): value for key, value in row.items()}
            identifier = _normalize_id(normalized_row.get("id", ""))
            if identifier in folds:
                raise FoldValidationError(f"Duplicate fold ID: {identifier!r}")
            try:
                fold = int(str(normalized_row.get("fold", "")).strip())
            except ValueError as exc:
                raise FoldValidationError(f"Invalid fold value for ID {identifier!r}") from exc
            if fold < 0 or fold >= n_folds:
                raise FoldValidationError(f"Fold {fold} is outside the configured range 0..{n_folds - 1}")
            folds[identifier] = fold
    missing = expected - folds.keys()
    extra = folds.keys() - expected
    if missing or extra:
        raise FoldValidationError(f"Fold ID mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    if oof_ids is not None and {_normalize_id(value) for value in oof_ids} != set(folds):
        raise FoldValidationError("Fold IDs do not match OOF prediction IDs")
    if min_valid_folds is not None:
        valid_fold_count = len(set(folds.values()))
        if valid_fold_count < min_valid_folds:
            raise FoldValidationError(
                f"Only {valid_fold_count} valid folds are present; "
                f"at least {min_valid_folds} are required"
            )
    if warn_small_fold_size:
        counts = {fold: list(folds.values()).count(fold) for fold in range(n_folds)}
        for fold, count in counts.items():
            if count < warn_small_fold_size:
                warnings.warn(f"Fold {fold} has only {count} rows", RuntimeWarning)
    return folds


def _normalize_id(value):
    identifier = str(value).strip()
    if not identifier:
        raise FoldValidationError("IDs must not be empty")
    return identifier


def _header(value):
    return str(value).strip()
