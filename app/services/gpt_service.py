import base64
import json
import os

from openai import AsyncOpenAI

from app.models.schemas import ElementDescription

ELEMENT_DESCRIPTION_PROMPT = """You're a very experient designer and I want you to describe for me all the layers you see in this design: I want you to return to me a json that is an array of the following type, relative to any kind of element, shapes, text, icons, images, any kind of visual elements:
{
  "type": "text" | "image" | "shape" | "background",
  "name": "A very intuitive name for the element",
  "description": "A good description about how, what and where is the element in the image"
}
The array of these elements should be ordered by z-index.

IMPORTANT: Return ONLY a valid JSON object with a single key "elements" containing the array. Example:
{"elements": [{"type": "background", "name": "...", "description": "..."}, ...]}"""


class GPTService:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self.model = "gpt-5.1"

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    async def describe_elements(self, image_data: bytes) -> list[ElementDescription]:
        """Analyze an image and return descriptions of all visual elements."""
        base64_image = base64.b64encode(image_data).decode("utf-8")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ELEMENT_DESCRIPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=4096,
        )

        content = response.choices[0].message.content
        if not content:
            return []

        data = json.loads(content)
        elements = data.get("elements", [])

        return [ElementDescription(**elem) for elem in elements]


gpt_service = GPTService()
