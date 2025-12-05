import base64
import json
import os

from openai import AsyncOpenAI

from app.models.schemas import ElementDescription

ELEMENT_DESCRIPTION_PROMPT = """
Imagine that every element in the image has it's own z-index. 

Return me a json in the following format with the element that has the biggest z-index, the one closest to the camera: 

{ 
	"type": "image" | "shape" | "background", 
	"name": "A very intuitive name for the element", 
	"description": "This description will be used to ask gemini to generate a maks over this element, so you should be very detailed when describing it to prevent gemini from painting undesired parts." 
}"""

ISOLATION_PROMPT_GENERATOR = """
I wanna tell gemini nano banana to paint completely as black the following object in the image and disappear with the rest of the image, letting it just a silhouette in a solid white background;

The prompt should be very precise to avoid gemini inventing new elements instead of painting the existing described one; 

You should be extremely rigid about keeping the black element in the exact position, size and shape, should be a perfect silhouette of the existing element;

Return me a json with the prompt ({{"prompt": ...}}): :

{element_name}:
{element_description}"""

REMOVAL_PROMPT_GENERATOR = """
I'm providing you two images:
1. The first image is the current design
2. The second image is a mask where the element to be removed is shown as a black silhouette on white background

I want you to generate a prompt for gemini nano banana telling it to remove the objects that would be under the mask if they were overlayed on top of the first image;

You should be extremely rigid about keeping the rest of the image intact, only removing and redrawing the area under the black silhouette;

The gemini doesn't have access to the mask and doesn't know the existance of it so you should be extremely detailed when describing what's under it, otherwise gemini may misunderstand and remove wrong elements;
"""


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

        # Handle single element response (new format)
        if "type" in data and "name" in data and "description" in data:
            return [ElementDescription(**data)]

        # Handle array response (old format with "elements" key)
        elements = data.get("elements", [])
        return [ElementDescription(**elem) for elem in elements]

    async def generate_isolation_prompt(
        self, element: ElementDescription, image_data: bytes
    ) -> str:
        """
        Generate a custom Gemini prompt for painting an element solid black.

        GPT creates a tailored prompt based on the element's name and description,
        along with the actual image for visual context, ensuring Gemini understands
        exactly what to paint black.
        """
        base64_image = base64.b64encode(image_data).decode("utf-8")

        prompt = ISOLATION_PROMPT_GENERATOR.format(
            element_name=element.name,
            element_description=element.description,
        )

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
            max_completion_tokens=512,
        )

        content = response.choices[0].message.content
        if not content:
            # Fallback to a basic prompt if GPT fails
            return f"Paint the {element.name} completely solid black, making it a silhouette."

        data = json.loads(content)
        return data.get("prompt", f"Paint the {element.name} completely solid black.")

    async def generate_removal_prompt(
        self, element: ElementDescription, image_data: bytes, mask_data: bytes
    ) -> str:
        """
        Generate a custom Gemini prompt for removing an element.

        GPT analyzes both images:
        - The current image state
        - The mask showing the element as black silhouette on white background

        And generates a detailed prompt for Gemini to remove what's under the mask.
        """
        base64_image = base64.b64encode(image_data).decode("utf-8")
        base64_mask = base64.b64encode(mask_data).decode("utf-8")

        # Add JSON instruction to the prompt
        prompt = REMOVAL_PROMPT_GENERATOR + '\n\nReturn me a json with the prompt ({"prompt": ...}):'

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
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_mask}",
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=512,
        )

        content = response.choices[0].message.content
        if not content:
            # Fallback to a basic prompt if GPT fails
            return f"Remove the {element.name} from the image and redraw the area."

        data = json.loads(content)
        return data.get("prompt", f"Remove the {element.name} from the image.")


gpt_service = GPTService()
