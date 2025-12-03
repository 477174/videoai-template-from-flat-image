from typing import Literal
from pydantic import BaseModel


class ElementDescription(BaseModel):
    """Element description returned by GPT."""

    type: Literal["text", "image", "shape", "background"]
    name: str
    description: str


class ExtractedElement(BaseModel):
    """Extracted element with position and image data."""

    id: str
    type: Literal["text", "image", "shape", "background"]
    name: str
    description: str
    x: int
    y: int
    width: int
    height: int
    src: str  # base64 data URI


class PolotnoPage(BaseModel):
    """A page in the Polotno design."""

    id: str
    children: list[ExtractedElement]


class PolotnoDesign(BaseModel):
    """Complete Polotno-compatible design JSON."""

    width: int
    height: int
    pages: list[PolotnoPage]


class ExtractionResult(BaseModel):
    """Internal result from image processing."""

    x: int
    y: int
    width: int
    height: int
    src: str  # base64 data URI
