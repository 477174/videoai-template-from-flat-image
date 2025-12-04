import base64
import json
import os

from openai import AsyncOpenAI

from app.models.schemas import ElementDescription

ELEMENT_DESCRIPTION_PROMPT = """
Describe very every single element you see in this image; 

You should return me a json of the format: 
{ 
	"elements": [ 
		{ 
			"type": "image" | "shape" | "background", 
			"name": "A very intuitive name for the element or composition", 
			"description": "This description should descript exact what composes the object, every element on it, to prevent misunderstanding when extracting it. You should describe perfectly where it is located too." 
		}, 
		... 
	] 
}"""

ISOLATION_PROMPT_GENERATOR = """
I wanna tell gemini nano banana to paint completely as black the following object in the image and disappear with the rest of the image, letting it just a silhouette in a solid white background;

The prompt should be very precise to avoid gemini inventing new elements instead of painting the existing described one; 

You should be rigid about modifying element's positions or adding new elements that doesn't exists in the image; return me a json with the prompt ({{"prompt": ...}}): :

{element_name}:
{element_description}"""

REMOVAL_PROMPT_GENERATOR = """
I wanna tell gemini nano banana to remove the following object from the image,
redrawing the area where it was to make it look like the element was never there;
the prompt should be very precise to avoid gemini inventing new elements instead of just removing the existing described one;
You should be rigid about modifying other element's positions or adding new elements that doesn't exists in the image;
return me a json with the prompt ({{"prompt": ...}}):

{element_name}:
{element_description}"""

UPDATE_REFERENCES_PROMPT = """
You should see this design like if it was a real life sight where one object was removed.
I need you to verify which objects from the list below are STILL VISIBLE in the current image.

The objects we expect to still have are:
{elements_json}

Please look at the current image carefully and:
1. REMOVE any object from the list that is NO LONGER VISIBLE in the image
2. UPDATE the "description" field for each remaining object to reflect its current appearance and location

IMPORTANT:
- If an object is not visible anymore, DO NOT include it in the response
- The "type" and "name" should remain the same for objects that are still visible
- Keep the ordering by distance from camera (far to near)
- The description should describe exactly what composes the object and where it is located
- Return ONLY a valid JSON object with a single key "elements" containing the array of VISIBLE objects only

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
        self, element: ElementDescription, image_data: bytes
    ) -> str:
        """
        Generate a custom Gemini prompt for creating a mask of an element.

        GPT creates a tailored prompt based on the element's name and description,
        along with the actual image for visual context, ensuring Gemini understands
        exactly what to paint black on a white background.
        """
        base64_image = base64.b64encode(image_data).decode("utf-8")

        prompt = REMOVAL_PROMPT_GENERATOR.format(
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
            return f"Remove the {element.name} from the image and redraw the area."

        data = json.loads(content)
        return data.get("prompt", f"Remove the {element.name} from the image.")

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

        # GPT may have removed elements that are no longer visible
        # This is expected behavior - don't validate count
        if not elements:
            # If no elements returned, something went wrong - return original
            return remaining_elements

        return [ElementDescription(**elem) for elem in elements]


gpt_service = GPTService()
