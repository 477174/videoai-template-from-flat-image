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
        3. GPT generates a custom prompt for Nano Banana to paint element black
        4. Use Nano Banana to create black silhouette with GPT's prompt
        5. Compare original vs silhouette to extract element pixels
        6. GPT generates a custom prompt for Nano Banana to remove the element
        7. Use Nano Banana to remove the element with GPT's prompt
        8. Repeat with the updated image until only 1 element remains
        9. The last element is the remaining image itself

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

            # Generate custom isolation prompt via GPT (with image context)
            isolation_prompt = await gpt_service.generate_isolation_prompt(
                current_element, image_state
            )
            debug.save_isolation_prompt(isolation_prompt, iter_dir)

            # Use GPT's prompt to create black silhouette
            black_image = await gemini_service.isolate_element(
                image_state, isolation_prompt
            )
            debug.save_black_isolated(black_image, iter_dir)

            # Compare original vs black to extract element pixels
            extracted = image_processor.extract_element(image_state, black_image)
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

            # Generate custom removal prompt via GPT (with image context)
            removal_prompt = await gpt_service.generate_removal_prompt(
                current_element, image_state
            )
            debug.save_removal_prompt(removal_prompt, iter_dir)

            # Remove element for next iteration using GPT's prompt
            image_state = await gemini_service.remove_element(
                image_state, removal_prompt
            )
            debug.save_after_removal(image_state, iter_dir)

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
