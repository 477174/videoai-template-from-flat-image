import base64
import io

import numpy as np
from PIL import Image
from scipy import ndimage

from app.models.schemas import ExtractionResult

# Threshold for detecting black pixels (0-255)
# Pixels with all RGB channels below this are considered "black" (part of element)
BLACK_THRESHOLD = 30

# Minimum size of connected region to keep (filters small noise/artifacts)
MIN_REGION_SIZE = 100


class ImageProcessor:
    """Service for pixel comparison and element extraction."""

    def extract_element(
        self, original_data: bytes, mask_data: bytes
    ) -> ExtractionResult:
        """
        Extract element pixels using a black-on-white mask.

        The mask image has:
        - Black pixels = the element (silhouette)
        - White pixels = background

        We find black pixels in the mask and extract those pixels from the original.
        """
        original = Image.open(io.BytesIO(original_data)).convert("RGBA")
        mask_image = Image.open(io.BytesIO(mask_data)).convert("RGB")

        if original.size != mask_image.size:
            mask_image = mask_image.resize(original.size, Image.Resampling.LANCZOS)

        original_array = np.array(original)
        mask_array = np.array(mask_image)

        # Find black pixels in the mask (element silhouette)
        # A pixel is "black" if all RGB channels are below threshold
        is_black = np.all(mask_array < BLACK_THRESHOLD, axis=2)

        # Clean up the mask
        mask = self._clean_mask(is_black)

        # Create result image with original pixels where mask is True
        result_array = np.zeros_like(original_array)
        result_array[mask] = original_array[mask]
        result_array[~mask, 3] = 0  # Set alpha to 0 for non-masked pixels

        result_image = Image.fromarray(result_array, "RGBA")

        return self._trim_and_encode(result_image)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up the binary mask using morphological operations.

        1. Remove small isolated regions (noise)
        2. Fill small holes
        """
        structure = ndimage.generate_binary_structure(2, 2)

        # Label connected regions
        labeled, num_features = ndimage.label(mask)

        if num_features == 0:
            return mask

        # Remove small regions (noise)
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_features + 1):
            region = labeled == i
            if np.sum(region) >= MIN_REGION_SIZE:
                cleaned_mask |= region

        # If all regions were too small, keep original
        if not np.any(cleaned_mask):
            return mask

        # Morphological closing to fill small holes
        cleaned_mask = ndimage.binary_closing(cleaned_mask, structure, iterations=2)

        return cleaned_mask

    def extract_full_image(self, image_data: bytes) -> ExtractionResult:
        """
        Convert the entire image to an extraction result.

        Used when only one element remains (the last element optimization).
        """
        image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        width, height = image.size

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return ExtractionResult(
            x=0,
            y=0,
            width=width,
            height=height,
            src=f"data:image/png;base64,{base64_data}",
        )

    def _trim_and_encode(self, image: Image.Image) -> ExtractionResult:
        """
        Remove transparent pixels from edges and encode as base64.

        Returns position and dimensions of the non-transparent region.
        """
        bbox = image.getbbox()

        if bbox is None:
            width, height = image.size
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return ExtractionResult(
                x=0,
                y=0,
                width=width,
                height=height,
                src=f"data:image/png;base64,{base64_data}",
            )

        x, y, x2, y2 = bbox
        width = x2 - x
        height = y2 - y

        cropped = image.crop(bbox)

        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return ExtractionResult(
            x=x,
            y=y,
            width=width,
            height=height,
            src=f"data:image/png;base64,{base64_data}",
        )

    def get_image_dimensions(self, image_data: bytes) -> tuple[int, int]:
        """Get the width and height of an image."""
        image = Image.open(io.BytesIO(image_data))
        return image.size


image_processor = ImageProcessor()
