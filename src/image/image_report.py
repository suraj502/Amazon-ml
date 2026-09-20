"""Image EDA and validation report helpers."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .image_duplicates import find_duplicate_images
from .image_preprocessing import get_image_metadata


def analyze_image_references(dataframe, image_column):
    """Analyze image references from a dataframe.

    Args:
        dataframe: Input pandas DataFrame.
        image_column: Column containing image paths or URLs.

    Returns:
        dict: Image availability and metadata summary.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if image_column not in dataframe.columns:
        raise ValueError(
            f"Image column {image_column!r} does not exist in the dataframe."
        )

    references = dataframe[image_column]

    total = len(references)
    missing = 0
    valid = 0
    invalid = 0
    urls = 0

    widths = []
    heights = []
    aspect_ratios = []
    formats = Counter()
    local_references = []

    for reference in references:
        metadata = get_image_metadata(reference)

        if metadata["kind"] == "missing":
            missing += 1
            continue

        if metadata["kind"] == "url":
            urls += 1
            continue

        if not metadata["valid"]:
            invalid += 1
            continue

        valid += 1
        local_references.append(metadata["reference"])

        if metadata["width"] is not None:
            widths.append(metadata["width"])

        if metadata["height"] is not None:
            heights.append(metadata["height"])

        if metadata["aspect_ratio"] is not None:
            aspect_ratios.append(metadata["aspect_ratio"])

        if metadata["format"]:
            formats[metadata["format"]] += 1

    duplicate_groups = find_duplicate_images(local_references)

    duplicate_image_count = sum(
        len(paths)
        for paths in duplicate_groups.values()
    )

    return {
        "total_rows": total,
        "missing_images": missing,
        "url_references": urls,
        "valid_images": valid,
        "invalid_images": invalid,
        "availability_rate": valid / total if total else 0.0,
        "dimensions": {
            "min_width": min(widths) if widths else None,
            "max_width": max(widths) if widths else None,
            "min_height": min(heights) if heights else None,
            "max_height": max(heights) if heights else None,
        },
        "aspect_ratio": {
            "min": min(aspect_ratios) if aspect_ratios else None,
            "max": max(aspect_ratios) if aspect_ratios else None,
        },
        "formats": dict(formats),
        "duplicates": {
            "duplicate_groups": duplicate_groups,
            "duplicate_image_count": duplicate_image_count,
        },
    }


def save_json_report(report, output_path):
    """Save an image report as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)


def save_markdown_report(report, output_path):
    """Save an image report as Markdown."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# M3 Image Report",
        "",
        "## Overview",
        "",
        f"- Total rows: {report['total_rows']}",
        f"- Valid local images: {report['valid_images']}",
        f"- Missing image references: {report['missing_images']}",
        f"- URL references: {report['url_references']}",
        f"- Invalid local images: {report['invalid_images']}",
        f"- Availability rate: {report['availability_rate']:.4f}",
        "",
        "## Dimensions",
        "",
        f"- Minimum width: {report['dimensions']['min_width']}",
        f"- Maximum width: {report['dimensions']['max_width']}",
        f"- Minimum height: {report['dimensions']['min_height']}",
        f"- Maximum height: {report['dimensions']['max_height']}",
        "",
        "## Aspect Ratio",
        "",
        f"- Minimum: {report['aspect_ratio']['min']}",
        f"- Maximum: {report['aspect_ratio']['max']}",
        "",
        "## Image Formats",
        "",
    ]

    if report["formats"]:
        for image_format, count in report["formats"].items():
            lines.append(f"- {image_format}: {count}")
    else:
        lines.append("- No readable local image formats detected.")

    lines.extend(
        [
            "",
            "## Duplicates",
            "",
            f"- Duplicate groups: "
            f"{len(report['duplicates']['duplicate_groups'])}",
            f"- Images belonging to duplicate groups: "
            f"{report['duplicates']['duplicate_image_count']}",
        ]
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def save_image_report(report, output_dir="reports"):
    """Save both JSON and Markdown versions of the M3 image report.

    Returns:
        tuple[Path, Path]: JSON and Markdown output paths.
    """
    output_dir = Path(output_dir)

    json_path = output_dir / "m3_image_report.json"
    markdown_path = output_dir / "m3_image_report.md"

    save_json_report(report, json_path)
    save_markdown_report(report, markdown_path)

    return json_path, markdown_path