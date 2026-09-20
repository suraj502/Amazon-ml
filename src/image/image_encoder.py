"""Pretrained image encoder utilities for M3."""

import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18ImageEncoder:
    """Frozen pretrained ResNet18 used as an image feature extractor."""

    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.weights = ResNet18_Weights.IMAGENET1K_V1

        self.model = resnet18(weights=self.weights)

        # Remove the classification head.
        self.model.fc = torch.nn.Identity()

        self.model.to(self.device)
        self.model.eval()

        # Freeze all encoder parameters.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.preprocess = self.weights.transforms()

    @property
    def embedding_dim(self):
        """Return the dimensionality of the extracted embedding."""
        return 512

    def encode(self, image):
        """Encode a single PIL image into a 512-dimensional vector."""
        image_tensor = self.preprocess(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(image_tensor)

        return embedding.squeeze(0).cpu().numpy()

    def encode_batch(self, images):
        """Encode a batch of PIL images into a 2D NumPy array."""
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        image_tensors = torch.stack(
            [self.preprocess(image) for image in images]
        ).to(self.device)

        with torch.no_grad():
            embeddings = self.model(image_tensors)

        return embeddings.cpu().numpy().astype(np.float32)