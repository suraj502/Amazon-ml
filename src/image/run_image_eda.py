"""Run the M3 image EDA pipeline."""

from pathlib import Path

import pandas as pd
import yaml

from .image_report import analyze_image_references, save_image_report


def load_config(config_path="config/config.yaml"):
    """Load the shared project configuration."""
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def find_image_column(dataframe, config):
    """Find the configured image column without guessing competition schema."""
    image_config = config.get("image", {})
    configured_columns = image_config.get("columns") or []

    if not configured_columns:
        return None

    for column in configured_columns:
        if column in dataframe.columns:
            return column

    return None


def run_image_eda(
    dataframe,
    config_path="config/config.yaml",
    output_dir="reports",
):
    """Run image EDA using the shared configuration."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    config = load_config(config_path)
    image_column = find_image_column(dataframe, config)

    if image_column is None:
        return {
            "status": "inactive",
            "reason": (
                "No configured image column is available in the dataset."
            ),
        }

    report = analyze_image_references(
        dataframe=dataframe,
        image_column=image_column,
    )

    json_path, markdown_path = save_image_report(
        report=report,
        output_dir=output_dir,
    )

    return {
        "status": "active",
        "image_column": image_column,
        "report": report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }