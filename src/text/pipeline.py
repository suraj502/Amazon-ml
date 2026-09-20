"""Dataset-independent M2 text preparation and prediction pipeline."""

from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.cv_split import validate_folds


_NUMBER = r"(?P<value>\d+(?:[.,]\d+)?)"
_NUMBER_PATTERN = r"\d+(?:[.,]\d+)?"
_UNIT = r"(?P<unit>[a-zA-Zµ]+)"
_UNIT_WORDS = {
    "kg", "g", "mg", "lb", "lbs", "oz", "ml", "l", "cl", "dl",
    "cm", "mm", "m", "in", "inch", "inches", "ft", "feet", "pcs",
    "piece", "pieces", "pack", "packs", "count", "ct", "x",
}


def normalize_id_series(values: pd.Series) -> pd.Series:
    """Normalize IDs without changing case or losing leading zeroes."""
    result = values.astype("string").str.strip()
    if result.isna().any() or (result == "").any():
        raise ValueError("IDs must be non-empty")
    if result.duplicated().any():
        raise ValueError("IDs must be unique")
    return result.astype(str)


def discover_text_columns(data: pd.DataFrame, config: dict) -> list[str]:
    """Use configured columns or discover string-like, non-ID/target columns."""
    text_config = config.get("text", {})
    configured = [str(column).strip() for column in text_config.get("columns", []) if str(column).strip()]
    excluded = {
        str(config["task"]["id_column"]).strip(),
        str(config["task"]["target"]).strip(),
    }
    if configured:
        missing = [column for column in configured if column not in data.columns]
        if missing:
            raise ValueError(f"Configured text columns are missing: {missing}")
        return configured

    discovered = [
        column for column in data.columns
        if column not in excluded
        and (
            pd.api.types.is_string_dtype(data[column])
            or pd.api.types.is_object_dtype(data[column])
        )
    ]
    if not discovered:
        raise ValueError(
            "No text columns were discovered. Set text.columns in config/config.yaml "
            "after inspecting the canonical dataset."
        )
    return discovered


def clean_text(value, config: dict) -> str:
    """Normalize one text value while retaining numbers and units."""
    normalization = config.get("text", {}).get("normalization", {})
    missing = config.get("text", {}).get("missing_text_value", "")
    if value is None or pd.isna(value):
        return str(missing)
    text = str(value)
    if normalization.get("remove_html", False):
        text = html.unescape(text)
        text = re.sub(r"<[^>]*>", " ", text)
    if normalization.get("unicode", False):
        text = unicodedata.normalize("NFKC", text)
    if normalization.get("lowercase", False):
        text = text.lower()
    if normalization.get("whitespace", False):
        text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_text(data: pd.DataFrame, text_columns: list[str], config: dict) -> pd.Series:
    """Combine and clean configured/discovered text columns."""
    return data[text_columns].apply(
        lambda row: " ".join(clean_text(value, config) for value in row),
        axis=1,
    )


def _number(value: str) -> float:
    return float(value.replace(",", "."))


def _first_match(pattern: str, text: str) -> tuple[float | str, str] | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return _number(match.group("value")), match.group("unit").lower()


def extract_structured_features(
    ids: pd.Series, text: pd.Series, config: dict
) -> pd.DataFrame:
    """Extract deterministic, label-free catalog attributes from text."""
    fields = config.get("text", {}).get("structured_extraction", {}).get("fields", {})
    rows = []
    for value in text.fillna("").astype(str):
        row = {}
        if fields.get("brand", {}).get("enabled", False):
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", value)
            row["brand"] = tokens[0] if tokens else ""
        if fields.get("product_type", {}).get("enabled", False):
            tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", value)
            row["product_type"] = " ".join(tokens[:3]) if tokens else ""
        if fields.get("pack_quantity", {}).get("enabled", False):
            match = re.search(
                rf"(?:pack(?:age)?|count|ct|pcs?|pieces?)\s*(?:of\s*)?{_NUMBER}",
                value,
                re.IGNORECASE,
            ) or re.search(rf"{_NUMBER}\s*(?:x|pack|packs|pcs?|pieces?|ct)\b", value, re.IGNORECASE)
            row["pack_quantity"] = _number(match.group("value")) if match else math.nan
        weight = _first_match(rf"{_NUMBER}\s*{_UNIT}", value)
        if fields.get("weight", {}).get("enabled", False):
            row["weight"] = weight[0] if weight and weight[1] in {"kg", "g", "mg", "lb", "lbs", "oz"} else math.nan
        if fields.get("weight_unit", {}).get("enabled", False):
            row["weight_unit"] = weight[1] if weight and weight[1] in {"kg", "g", "mg", "lb", "lbs", "oz"} else ""
        if fields.get("volume", {}).get("enabled", False):
            row["volume"] = weight[0] if weight and weight[1] in {"ml", "l", "cl", "dl"} else math.nan
        if fields.get("volume_unit", {}).get("enabled", False):
            row["volume_unit"] = weight[1] if weight and weight[1] in {"ml", "l", "cl", "dl"} else ""
        if fields.get("dimensions", {}).get("enabled", False):
            dimensions = re.search(
                rf"{_NUMBER_PATTERN}\s*[x×]\s*{_NUMBER_PATTERN}"
                rf"(?:\s*[x×]\s*{_NUMBER_PATTERN})?\s*{_UNIT}?",
                value,
                re.IGNORECASE,
            )
            row["dimensions"] = dimensions.group(0) if dimensions else ""
        if fields.get("numeric_count", {}).get("enabled", False):
            row["numeric_count"] = len(re.findall(r"\d+(?:[.,]\d+)?", value))
        if fields.get("unit_count", {}).get("enabled", False):
            row["unit_count"] = sum(
                1 for token in re.findall(r"[A-Za-zµ]+", value.lower()) if token in _UNIT_WORDS
            )
        if fields.get("model_number", {}).get("enabled", False):
            model = re.search(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{2,}\b", value)
            row["model_number"] = model.group(0) if model else ""
        rows.append(row)
    return pd.concat(
        [ids.reset_index(drop=True).rename(str(config["task"]["id_column"])),
         pd.DataFrame(rows)],
        axis=1,
    )


def write_structured_features(features: pd.DataFrame, path: str | Path, overwrite: bool = False) -> None:
    """Write structured features without silently replacing an existing artifact."""
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing feature file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)


def load_dataset(path: str | Path, config: dict) -> pd.DataFrame:
    """Load a configured CSV with headers normalized as configured."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {input_path}")
    id_column = config.get("task", {}).get("id_column")
    dtype = {id_column: str} if id_column else None
    data = pd.read_csv(
        input_path,
        encoding=config["data"].get("encoding", "utf-8-sig"),
        dtype=dtype,
    )
    if config["data"].get("strip_header_whitespace", True):
        data.columns = [str(column).strip() for column in data.columns]
    return data


def write_report(report: dict, json_path: str | Path, markdown_path: str | Path) -> None:
    """Write EDA metadata only after a real dataset has been supplied."""
    json_file, markdown_file = Path(json_path), Path(markdown_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = ["# M2 text report", ""]
    lines.extend(f"- **{key}:** {value}" for key, value in report.items())
    markdown_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_shared_folds(config: dict, train_ids: pd.Series) -> dict[str, int]:
    """Validate and return M1's fold assignment; never generate folds."""
    folds_path = Path(config["validation"]["folds_path"])
    return validate_folds(
        folds_path,
        train_ids,
        config["validation"]["n_folds"],
        warn_small_fold_size=config["validation"].get("warn_small_fold_size"),
    )
