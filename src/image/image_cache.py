"""Utilities for deterministic image caching and resume-safe processing."""

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse


def build_cache_key(reference):
    """Build a deterministic SHA-256 cache key from an image reference."""
    if reference is None:
        raise ValueError("Image reference cannot be None.")

    value = str(reference).strip()
    if not value:
        raise ValueError("Image reference cannot be empty.")

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_cache_path(reference, cache_dir="data/image_cache"):
    """Return the deterministic cache path for an image reference."""
    cache_dir = Path(cache_dir)
    cache_key = build_cache_key(reference)

    parsed = urlparse(str(reference).strip())
    suffix = Path(parsed.path).suffix.lower()

    if not suffix or len(suffix) > 10:
        suffix = ".img"

    return cache_dir / f"{cache_key}{suffix}"


def get_metadata_path(reference, cache_dir="data/image_cache"):
    """Return the metadata path associated with a cached image."""
    cache_path = get_cache_path(reference, cache_dir)
    return cache_path.with_suffix(".json")


def is_cached(reference, cache_dir="data/image_cache"):
    """Return True when the cached image already exists."""
    return get_cache_path(reference, cache_dir).is_file()


def save_cache_metadata(reference, metadata, cache_dir="data/image_cache"):
    """Save metadata for a cached image."""
    metadata_path = get_metadata_path(reference, cache_dir)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "reference": str(reference).strip(),
        "cache_key": build_cache_key(reference),
        **metadata,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return metadata_path


def load_cache_metadata(reference, cache_dir="data/image_cache"):
    """Load cached metadata if it exists."""
    metadata_path = get_metadata_path(reference, cache_dir)

    if not metadata_path.is_file():
        return None

    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_cache_complete(reference, cache_dir="data/image_cache"):
    """Return True when both image and metadata exist."""
    return (
        is_cached(reference, cache_dir)
        and get_metadata_path(reference, cache_dir).is_file()
    )