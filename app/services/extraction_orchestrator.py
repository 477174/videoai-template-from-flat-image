import uuid

from app.models.schemas import ExtractedElement
from app.services.debug_saver import DebugSaver
from app.services.gemini_service import gemini_service
from app.services.gpt_service import gpt_service
from app.services.image_processor import image_processor


class ExtractionOrchestrator:
    """Orchestrates the iterative element extraction process."""

    async def extract_all_elements(
        self, original_image: bytes, filename: str = "image"
    ) -> list[ExtractedElement]:
        """
        Extract all elements from an image using iterative processing.

        Process flow:
        1. Analyze image with GPT to get element descriptions (ordered by z-index)
        2. Pick the last element (top z-index)
        3. Use Nano Banana to remove the element from the image
        4. Compare before/after to extract the element pixels
        5. Repeat with the updated image until only 1 element remains
        6. The last element is the remaining image itself

        Returns elements in z-index order (bottom to top).
        """
        debug = DebugSaver(filename)
        debug.save_original(original_image)

        image_state = original_image
        extracted_elements: list[ExtractedElement] = []
        iteration = 1

        # Initial element description - only done once
        elements_description = await gpt_service.describe_elements(image_state)

        if not elements_description:
            return []

        debug.save_elements_description(elements_description, iteration)

        while True:
            if len(elements_description) == 1:
                iter_dir = debug.start_iteration(iteration)
                debug.save_image_state(image_state, iter_dir)

                last_element = elements_description[0]
                debug.save_element_info(last_element, iter_dir)

                extracted = image_processor.extract_full_image(image_state)
                debug.save_extracted_element(extracted, iter_dir)

                extracted_elements.append(
                    ExtractedElement(
                        id=str(uuid.uuid4()),
                        type=last_element.type,
                        name=last_element.name,
                        description=last_element.description,
                        x=extracted.x,
                        y=extracted.y,
                        width=extracted.width,
                        height=extracted.height,
                        src=extracted.src,
                    )
                )
                break

            iter_dir = debug.start_iteration(iteration)
            debug.save_image_state(image_state, iter_dir)

            current_element = elements_description[-1]
            debug.save_element_info(current_element, iter_dir)

            # Remove element and get new image state (single API call)
            image_after_removal = await gemini_service.remove_element(
                image_state, current_element
            )
            debug.save_after_removal(image_after_removal, iter_dir)

            # Compare before/after to extract the element pixels
            extracted = image_processor.extract_element(
                image_state, image_after_removal
            )
            debug.save_extracted_element(extracted, iter_dir)

            extracted_elements.append(
                ExtractedElement(
                    id=str(uuid.uuid4()),
                    type=current_element.type,
                    name=current_element.name,
                    description=current_element.description,
                    x=extracted.x,
                    y=extracted.y,
                    width=extracted.width,
                    height=extracted.height,
                    src=extracted.src,
                )
            )

            # Update image state for next iteration
            image_state = image_after_removal

            # Remove the processed element from the list
            remaining_elements = elements_description[:-1]

            if not remaining_elements:
                break

            # Update references for remaining elements based on new image state
            elements_description = await gpt_service.update_element_references(
                image_state, remaining_elements
            )

            iteration += 1
            debug.save_elements_description(elements_description, iteration)

        result = list(reversed(extracted_elements))

        debug.save_final_result({
            "elements": [elem.model_dump() for elem in result]
        })

        return result


extraction_orchestrator = ExtractionOrchestrator()
