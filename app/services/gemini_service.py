import asyncio
import io
import os

from google import genai
from google.genai import types
from PIL import Image

from app.models.schemas import ElementDescription

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


class GeminiService:
    """Service for interacting with Nano Banana (Gemini) image editing API."""

    def __init__(self) -> None:
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return self._client

    async def isolate_element(
        self, image_data: bytes, element: ElementDescription
    ) -> bytes:
        """Make the specified element black (saturation 0, lightness 0, alpha 100%)."""
        prompt = f"""In this design I want you to change the colors of the {element.name} ({element.description}) in the following way:

Set saturation 0;
Set lightness 0;
Set alpha 100%;"""

        return await self._edit_image(image_data, prompt)

    async def remove_element(
        self, image_data: bytes, element: ElementDescription
    ) -> bytes:
        """Remove the specified element from the image."""
        prompt = f"From this image, remove the {element.name} ({element.description})"

        return await self._edit_image(image_data, prompt)

    async def _edit_image(self, image_data: bytes, prompt: str) -> bytes:
        """Send image editing request to Gemini API with retry logic."""
        image = Image.open(io.BytesIO(image_data))

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.aio.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                    ),
                )

                # Handle null response cases
                if not response.candidates:
                    raise ValueError("Gemini API returned no candidates")

                candidate = response.candidates[0]
                if candidate.content is None:
                    finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
                    raise ValueError(
                        f"Gemini API returned no content. Finish reason: {finish_reason}"
                    )

                for part in candidate.content.parts:
                    if part.inline_data is not None:
                        return part.inline_data.data

                raise ValueError("No image returned from Gemini API")

            except ValueError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff
                    delay = RETRY_DELAY * (attempt + 1)
                    await asyncio.sleep(delay)
                    continue
                raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise ValueError("Gemini API failed after all retries")


gemini_service = GeminiService()
