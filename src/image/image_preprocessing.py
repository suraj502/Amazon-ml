"""Image reference validation and safe image loading utilities.

This module does not assume a specific competition image column or source.
It provides reusable helpers for validating and loading image references.
"""

from pathlib import Path
from urllib.parse import urlparse

SUPPORTED_URL_SCHEMES = {"http", "https"}


class ImageValidationError(ValueError):
    """Raised when an image reference is invalid."""


def validate_image_reference(reference):
    """Validate an image reference without loading the image.

    Supported references:
    - local filesystem paths
    - HTTP/HTTPS URLs

    Returns:
        dict: Basic information about the reference.
    """
    if reference is None:
        return {
            "valid": False,
            "kind": "missing",
            "reference": None,
            "reason": "Image reference is missing.",
        }

    value = str(reference).strip()

    if not value:
        return {
            "valid": False,
            "kind": "missing",
            "reference": value,
            "reason": "Image reference is empty.",
        }

    parsed = urlparse(value)

    if parsed.scheme in SUPPORTED_URL_SCHEMES:
        return {
            "valid": True,
            "kind": "url",
            "reference": value,
            "reason": None,
        }

    path = Path(value)

    if path.is_file():
        return {
            "valid": True,
            "kind": "local",
            "reference": value,
            "reason": None,
        }

    return {
        "valid": False,
        "kind": "local",
        "reference": value,
        "reason": "Local image file does not exist.",
    }


def load_image(reference):
    """Safely load a local image.

    URL downloading is intentionally not implemented yet because the
    competition image source is not known.

    Returns:
        PIL.Image.Image: Loaded RGB image.

    Raises:
        ImageValidationError: If the reference is missing, unsupported,
            or the image cannot be opened.
    """
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ImageValidationError("Pillow is required to load images. Install requirements.txt.") from exc

    validation = validate_image_reference(reference)

    if not validation["valid"]:
        raise ImageValidationError(validation["reason"])

    if validation["kind"] == "url":
        raise ImageValidationError(
            "URL loading is not implemented until the competition image "
            "source and download policy are known."
        )

    try:
        with Image.open(validation["reference"]) as image:
            image.load()
            return image.convert("RGB")

    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(
            f"Unable to read image: {validation['reference']}"
        ) from exc


def get_image_metadata(reference):
    """Read basic metadata from a local image.

    Returns:
        dict: Validation status and image metadata.
    """
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ImageValidationError("Pillow is required to inspect images. Install requirements.txt.") from exc

    validation = validate_image_reference(reference)

    result = {
        "reference": validation["reference"],
        "kind": validation["kind"],
        "valid": validation["valid"],
        "reason": validation["reason"],
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "format": None,
        "mode": None,
        "file_size_bytes": None,
    }

    if not validation["valid"] or validation["kind"] != "local":
        return result

    try:
        with Image.open(validation["reference"]) as image:
            width, height = image.size

            result.update(
                {
                    "valid": True,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height if height else None,
                    "format": image.format,
                    "mode": image.mode,
                    "file_size_bytes": Path(
                        validation["reference"]
                    ).stat().st_size,
                }
            )

    except (UnidentifiedImageError, OSError) as exc:
        result.update(
            {
                "valid": False,
                "reason": f"Unable to read image: {exc}",
            }
        )

    return result


def preprocess_image(image, image_size=224):
    """Prepare a PIL image for a future vision model.

    The actual model-specific preprocessing will be added later.
    For now this performs only deterministic resizing and RGB conversion.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImageValidationError("Pillow is required to preprocess images. Install requirements.txt.") from exc

    if not isinstance(image, Image.Image):
        raise ImageValidationError("Expected a PIL Image object.")

    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer.")

    image = image.convert("RGB")

    return image.resize((image_size, image_size))