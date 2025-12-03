import uuid

from app.models.schemas import ExtractedElement, PolotnoDesign, PolotnoPage


class PolotnoFormatter:
    """Formats extracted elements into Polotno-compatible JSON."""

    def format_design(
        self,
        elements: list[ExtractedElement],
        width: int,
        height: int,
    ) -> PolotnoDesign:
        """
        Create a Polotno-compatible design JSON from extracted elements.

        Args:
            elements: List of extracted elements in z-index order (bottom to top)
            width: Original image width
            height: Original image height

        Returns:
            PolotnoDesign with a single page containing all elements
        """
        page = PolotnoPage(
            id=str(uuid.uuid4()),
            children=elements,
        )

        return PolotnoDesign(
            width=width,
            height=height,
            pages=[page],
        )


polotno_formatter = PolotnoFormatter()
