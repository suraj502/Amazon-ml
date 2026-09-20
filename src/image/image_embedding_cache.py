"""Disk-backed image embedding cache utilities."""

from pathlib import Path

import numpy as np

from .image_cache import build_cache_key


class ImageEmbeddingCache:
    """Store and retrieve image embeddings using deterministic cache keys."""

    def __init__(self, cache_dir="features/image_embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, reference):
        """Return the embedding path for an image reference."""
        cache_key = build_cache_key(reference)
        return self.cache_dir / f"{cache_key}.npy"

    def exists(self, reference):
        """Check whether an embedding is already cached."""
        return self.get_path(reference).is_file()

    def save(self, reference, embedding):
        """Save an embedding to disk."""
        embedding = np.asarray(embedding, dtype=np.float32)

        path = self.get_path(reference)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.save(path, embedding)

        return path

    def load(self, reference):
        """Load a cached embedding."""
        path = self.get_path(reference)

        if not path.is_file():
            raise FileNotFoundError(
                f"No cached embedding found for: {reference}"
            )

        return np.load(path)

    def get_or_none(self, reference):
        """Load an embedding if cached, otherwise return None."""
        if not self.exists(reference):
            return None

        return self.load(reference)