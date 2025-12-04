import base64
import io

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

from app.models.schemas import ExtractionResult

# SSIM threshold - lower = less sensitive to minor changes
# When comparing original vs black silhouette, the change is dramatic
# so we can use a lower threshold to filter out compression artifacts
SSIM_THRESHOLD = 0.70

# Gaussian blur sigma to smooth artifacts before comparison
BLUR_SIGMA = 1.5

# Minimum size of connected region to keep (filters small noise/artifacts)
MIN_REGION_SIZE = 200


class ImageProcessor:
    """Service for pixel comparison and element extraction."""

    def extract_element(
        self, original_data: bytes, modified_data: bytes
    ) -> ExtractionResult:
        """
        Compare original and modified images using SSIM to extract changed pixels.

        Uses Structural Similarity Index (SSIM) which is robust to compression
        artifacts. When comparing original vs black silhouette, the dramatic
        color change makes detection reliable.
        """
        original = Image.open(io.BytesIO(original_data)).convert("RGBA")
        modified = Image.open(io.BytesIO(modified_data)).convert("RGBA")

        if original.size != modified.size:
            modified = modified.resize(original.size, Image.Resampling.LANCZOS)

        original_array = np.array(original)
        modified_array = np.array(modified)

        # Use RGB channels for SSIM comparison (ignore alpha)
        original_rgb = original_array[:, :, :3].astype(np.float64)
        modified_rgb = modified_array[:, :, :3].astype(np.float64)

        # Apply Gaussian blur to reduce artifact sensitivity
        original_blurred = gaussian_filter(original_rgb, sigma=BLUR_SIGMA)
        modified_blurred = gaussian_filter(modified_rgb, sigma=BLUR_SIGMA)

        # Calculate SSIM with full difference image on blurred images
        # Returns per-pixel similarity scores (0-1, where 1 = identical)
        _, diff = ssim(
            original_blurred,
            modified_blurred,
            full=True,
            channel_axis=2,
            data_range=255,
        )

        # Convert to single channel (average across RGB)
        diff_gray = np.mean(diff, axis=2)

        # Pixels with low similarity = changed pixels (element painted black)
        mask = diff_gray < SSIM_THRESHOLD

        # Clean up the mask with morphological operations
        mask = self._clean_mask(mask)

        # Create result image with original pixels where mask is True
        result_array = np.zeros_like(original_array)
        result_array[mask] = original_array[mask]
        result_array[~mask, 3] = 0  # Set alpha to 0 for non-masked pixels

        result_image = Image.fromarray(result_array, "RGBA")

        return self._trim_and_encode(result_image)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up the binary mask using morphological operations.

        1. Erode to shrink artifact pixels at edges
        2. Remove small isolated regions (noise)
        3. Fill small holes
        4. Smooth edges
        """
        structure = ndimage.generate_binary_structure(2, 2)

        # Initial erosion to shrink artifact pixels
        eroded_mask = ndimage.binary_erosion(mask, structure, iterations=2)

        # Label connected regions
        labeled, num_features = ndimage.label(eroded_mask)

        if num_features == 0:
            return mask  # Return original if erosion removed everything

        # Remove small regions
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_features + 1):
            region = labeled == i
            if np.sum(region) >= MIN_REGION_SIZE:
                cleaned_mask |= region

        # If all regions were too small, try with original mask
        if not np.any(cleaned_mask):
            labeled, num_features = ndimage.label(mask)
            for i in range(1, num_features + 1):
                region = labeled == i
                if np.sum(region) >= MIN_REGION_SIZE // 2:
                    cleaned_mask |= region

        # Morphological closing to fill small holes (more iterations)
        cleaned_mask = ndimage.binary_closing(cleaned_mask, structure, iterations=3)

        # Dilate back to restore size after initial erosion
        cleaned_mask = ndimage.binary_dilation(cleaned_mask, structure, iterations=2)

        # Final opening to smooth edges
        cleaned_mask = ndimage.binary_opening(cleaned_mask, structure, iterations=1)

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
