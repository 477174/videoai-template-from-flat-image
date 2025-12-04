import io
import json
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

from app.models.schemas import ElementDescription, ExtractionResult

DEBUG_DIR = Path(__file__).resolve().parent.parent.parent / "debug"


class DebugSaver:
    """Saves intermediate states during extraction for debugging."""

    def __init__(self, original_filename: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = re.sub(r"[^\w\-.]", "_", original_filename)
        self.session_dir = DEBUG_DIR / f"{timestamp}_{safe_filename}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.iteration = 0

    def save_original(self, image_data: bytes) -> None:
        """Save the original input image."""
        self._save_image(image_data, self.session_dir / "00_original.png")

    def save_elements_description(
        self, elements: list[ElementDescription], iteration: int
    ) -> None:
        """Save the GPT elements description for current iteration."""
        data = [elem.model_dump() for elem in elements]
        filename = f"{iteration:02d}_elements_description.json"
        self._save_json(data, self.session_dir / filename)

    def start_iteration(self, iteration: int) -> Path:
        """Create directory for a new iteration and return its path."""
        self.iteration = iteration
        iter_dir = self.session_dir / f"{iteration:02d}_iteration"
        iter_dir.mkdir(exist_ok=True)
        return iter_dir

    def save_image_state(self, image_data: bytes, iter_dir: Path) -> None:
        """Save current image state at start of iteration."""
        self._save_image(image_data, iter_dir / "01_image_state.png")

    def save_element_info(
        self, element: ElementDescription, iter_dir: Path
    ) -> None:
        """Save info about element being processed."""
        self._save_json(element.model_dump(), iter_dir / "02_element_info.json")

    def save_after_removal(self, image_data: bytes, iter_dir: Path) -> None:
        """Save image after element removal."""
        self._save_image(image_data, iter_dir / "03_after_removal.png")

    def save_extracted_element(
        self, result: ExtractionResult, iter_dir: Path
    ) -> None:
        """Save the extracted element image and its metadata."""
        import base64

        src = result.src
        if src.startswith("data:image/png;base64,"):
            src = src[22:]

        image_data = base64.b64decode(src)
        self._save_image(image_data, iter_dir / "04_extracted_element.png")

        metadata = {
            "x": result.x,
            "y": result.y,
            "width": result.width,
            "height": result.height,
        }
        self._save_json(metadata, iter_dir / "04_extracted_metadata.json")

    def save_final_result(self, result: dict) -> None:
        """Save the final Polotno JSON result."""
        self._save_json(result, self.session_dir / "final_result.json")

    def _save_image(self, image_data: bytes, path: Path) -> None:
        """Save image bytes to file."""
        image = Image.open(io.BytesIO(image_data))
        image.save(path, "PNG")

    def _save_json(self, data: dict | list, path: Path) -> None:
        """Save JSON data to file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
