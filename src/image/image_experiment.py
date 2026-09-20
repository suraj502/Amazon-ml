"""Experiment logging utilities for the M3 image pipeline."""

from pathlib import Path

import pandas as pd


EXPERIMENT_COLUMNS = [
    "experiment",
    "model",
    "features",
    "folds",
    "alpha",
    "status",
    "notes",
]


def create_experiment_record(
    experiment,
    model,
    features,
    folds,
    alpha,
    status,
    notes,
):
    """Create one standardized M3 experiment record."""

    return {
        "experiment": experiment,
        "model": model,
        "features": features,
        "folds": folds,
        "alpha": alpha,
        "status": status,
        "notes": notes,
    }


def save_experiment_record(
    record,
    output_path="reports/m3_experiments.csv",
):
    """Append an experiment record to the M3 experiment log."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame([record], columns=EXPERIMENT_COLUMNS)

    if output_path.exists():
        existing = pd.read_csv(output_path)

        if existing.columns.tolist() != EXPERIMENT_COLUMNS:
            raise ValueError(
                "Existing experiment log has an incompatible schema."
            )

        dataframe = pd.concat(
            [existing, dataframe],
            ignore_index=True,
        )

    dataframe.to_csv(output_path, index=False)

    return output_path