from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import PolotnoDesign
from app.services.extraction_orchestrator import extraction_orchestrator
from app.services.image_processor import image_processor
from app.services.polotno_formatter import polotno_formatter

router = APIRouter()


ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
}


@router.post("/api/extract", response_model=PolotnoDesign)
async def extract_elements(file: UploadFile = File(...)) -> PolotnoDesign:
    """
    Extract design elements from an uploaded image.

    Returns a Polotno-compatible JSON with all extracted elements.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    image_data = await file.read()

    if not image_data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    width, height = image_processor.get_image_dimensions(image_data)

    filename = file.filename or "image"
    elements = await extraction_orchestrator.extract_all_elements(image_data, filename)

    design = polotno_formatter.format_design(elements, width, height)

    return design
