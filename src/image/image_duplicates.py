"""Utilities for detecting duplicate image files."""

import hashlib
from pathlib import Path


def calculate_file_hash(path, chunk_size=1024 * 1024):
    """Calculate a SHA-256 hash for a local file."""
    path = Path(path)

    if not path.is_file():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def find_duplicate_images(references):
    """Find duplicate local files based on SHA-256 content hashes.

    Args:
        references: Iterable of local image paths.

    Returns:
        dict mapping file hashes to lists of duplicate paths.
        Only hashes with more than one path are returned.
    """
    hashes = {}

    for reference in references:
        if reference is None:
            continue

        path = Path(str(reference).strip())

        if not path.is_file():
            continue

        file_hash = calculate_file_hash(path)

        if file_hash is None:
            continue

        hashes.setdefault(file_hash, []).append(str(path))

    return {
        file_hash: paths
        for file_hash, paths in hashes.items()
        if len(paths) > 1
    }