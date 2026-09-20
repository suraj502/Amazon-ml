"""DataFrame integration for the M3 image embedding pipeline."""

import numpy as np
import pandas as pd

from .image_embedding_pipeline import ImageEmbeddingPipeline


def build_image_embeddings(
    dataframe,
    id_column,
    image_column,
    pipeline=None,
    batch_size=32,
):
    """Build image embeddings while preserving dataset IDs and row order.

    Args:
        dataframe: Input dataset.
        id_column: Original dataset ID column.
        image_column: Column containing image references.
        pipeline: Optional ImageEmbeddingPipeline instance.
        batch_size: Number of uncached images processed per batch.

    Returns:
        tuple[pd.DataFrame, list]:
            Embedding DataFrame and image processing statuses.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if id_column not in dataframe.columns:
        raise ValueError(
            f"ID column {id_column!r} does not exist in the dataframe."
        )

    if image_column not in dataframe.columns:
        raise ValueError(
            f"Image column {image_column!r} does not exist in the dataframe."
        )

    if dataframe[id_column].isna().any():
        raise ValueError("ID column contains missing values.")

    if dataframe[id_column].duplicated().any():
        raise ValueError("ID column contains duplicate values.")

    pipeline = pipeline or ImageEmbeddingPipeline()

    embeddings, statuses = pipeline.process(
        dataframe[image_column].tolist(),
        batch_size=batch_size,
    )

    embedding_columns = [
        f"embedding_{index}"
        for index in range(embeddings.shape[1])
    ]

    embedding_dataframe = pd.DataFrame(
        embeddings,
        columns=embedding_columns,
        index=dataframe.index,
    )

    result = pd.concat(
        [
            dataframe[[id_column]].copy(),
            embedding_dataframe,
        ],
        axis=1,
    )

    return result, statuses