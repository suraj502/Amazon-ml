"""Reusable batch image embedding extraction pipeline."""

import numpy as np


from .image_embedding_cache import ImageEmbeddingCache
from .image_encoder import ResNet18ImageEncoder
from .image_preprocessing import ImageValidationError, load_image


class ImageEmbeddingPipeline:
    """Extract and cache image embeddings in a resume-safe manner."""

    def __init__(
        self,
        embedding_cache=None,
        encoder=None,
        cache_dir="features/image_embeddings",
    ):
        self.cache = embedding_cache or ImageEmbeddingCache(cache_dir)
        self.encoder = encoder or ResNet18ImageEncoder()

    def process(self, references, batch_size=32):
        """Process image references and return embeddings plus status.

        Args:
            references: Iterable of image references.
            batch_size: Number of uncached images processed at once.

        Returns:
            tuple:
                embeddings: NumPy array of shape (n, embedding_dim).
                statuses: List of per-image processing statuses.
        """
        references = list(references)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        embeddings = np.zeros(
            (len(references), self.encoder.embedding_dim),
            dtype=np.float32,
        )

        statuses = []

        pending_images = []
        pending_indices = []

        for index, reference in enumerate(references):
            if reference is None or not str(reference).strip():
                statuses.append(
                    {
                        "status": "missing",
                        "reference": reference,
                    }
                )
                continue

            if self.cache.exists(reference):
                embeddings[index] = self.cache.load(reference)

                statuses.append(
                    {
                        "status": "cached",
                        "reference": reference,
                    }
                )
                continue

            try:
                image = load_image(reference)

                pending_images.append(image)
                pending_indices.append(index)

                statuses.append(
                    {
                        "status": "pending",
                        "reference": reference,
                    }
                )

            except ImageValidationError as exc:
                statuses.append(
                    {
                        "status": "invalid",
                        "reference": reference,
                        "reason": str(exc),
                    }
                )

        for start in range(0, len(pending_images), batch_size):
            batch_images = pending_images[start:start + batch_size]
            batch_indices = pending_indices[start:start + batch_size]

            batch_embeddings = self.encoder.encode_batch(batch_images)

            for index, reference, embedding in zip(
                batch_indices,
                [references[i] for i in batch_indices],
                batch_embeddings,
            ):
                self.cache.save(reference, embedding)
                embeddings[index] = embedding

                statuses[index]["status"] = "computed"

        return embeddings, statuses