import base64
import json
import os

from openai import AsyncOpenAI

from app.models.schemas import ElementDescription

ELEMENT_DESCRIPTION_PROMPT = """
You're a very experient designer and I want you to describe for me all the layers you see in this design:
I want you to return to me a json containing all images that was used in the construction of this design.
You can't be that granular, the elements or group of elements you're going to describe should be chosen by them semantic role in the design.
Before choosing an element or a group of elements you should think "is this something relevant for a regular user to be able to edit later?".


{
  "type": "image" | "shape" | "background",
  "name": "A very intuitive name for the element",
  "description": "This description should be very clear about what and where is the element in the image, should detail exactly what belongs to the element and what not, so later the element can be identified easily in the image. It should include position, colors, size, relation to other elements, etc."
}

The array of these elements should be ordered by z-index.

IMPORTANT: Return ONLY a valid JSON object with a single key "elements" containing the array. Example:
{"elements": [{"type": "background", "name": "...", "description": "..."}, ...]}"""

UPDATE_REFERENCES_PROMPT = """
You're a very experient designer. I have a design where one element was removed.
I need you to update the descriptions of the remaining elements based on how they now appear in the image.

The elements we still have (that were NOT removed) are:
{elements_json}

Please look at the current image and update the "description" field for each element to accurately reflect its current position and appearance in the image. The "type" and "name" should remain the same.

IMPORTANT: Return ONLY a valid JSON object with a single key "elements" containing the updated array. Keep the same order.
Example: {{"elements": [{{"type": "background", "name": "...", "description": "updated description..."}}, ...]}}"""


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

    async def update_element_references(
        self, image_data: bytes, remaining_elements: list[ElementDescription]
    ) -> list[ElementDescription]:
        """
        Update element descriptions based on the current image state.

        After an element is removed, the remaining elements may have shifted
        or their visual context changed. This method asks GPT to update the
        descriptions to match the current image.
        """
        base64_image = base64.b64encode(image_data).decode("utf-8")

        # Convert elements to JSON for the prompt
        elements_json = json.dumps(
            [elem.model_dump() for elem in remaining_elements],
            indent=2,
        )

        prompt = UPDATE_REFERENCES_PROMPT.format(elements_json=elements_json)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
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
            # If update fails, return original elements unchanged
            return remaining_elements

        data = json.loads(content)
        elements = data.get("elements", [])

        # Validate we got the same number of elements back
        if len(elements) != len(remaining_elements):
            # If mismatch, return original elements
            return remaining_elements

        return [ElementDescription(**elem) for elem in elements]


gpt_service = GPTService()
