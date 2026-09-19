from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select

from app.db import get_session
from app.models import Item, ReferenceImage, Search, SearchStatus
from app.storage import InvalidImageError, save_reference_image

router = APIRouter(prefix="/api")

MAX_REFERENCE_IMAGES = 3


class SearchCreateResponse(BaseModel):
    search_id: str
    item_id: str
    target_text: str
    reference_count: int
    status: str


@router.post("/searches", response_model=SearchCreateResponse)
async def create_search(
    target_text: str = Form(...),
    reference_images: list[UploadFile] = File(default=[]),
) -> SearchCreateResponse:
    text = target_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="target_text must not be empty")

    if len(reference_images) > MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=400, detail=f"at most {MAX_REFERENCE_IMAGES} reference images are allowed"
        )

    normalized = text.lower()

    with get_session() as session:
        item = session.exec(select(Item).where(func.lower(Item.name) == normalized)).first()

        if item is None:
            item = Item(name=text)
            session.add(item)

        existing_images = session.exec(
            select(ReferenceImage).where(ReferenceImage.item_id == item.id)
        ).all()
        remaining_slots = max(0, MAX_REFERENCE_IMAGES - len(existing_images))

        saved = 0
        for upload in reference_images[:remaining_slots]:
            if not (upload.content_type or "").startswith("image/"):
                raise HTTPException(status_code=400, detail=f"{upload.filename} is not an image")
            try:
                relative_path = await save_reference_image(item.id, upload)
            except InvalidImageError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            session.add(ReferenceImage(item_id=item.id, image_path=relative_path))
            saved += 1

        search = Search(item_id=item.id, target_text=text, status=SearchStatus.SEARCHING)
        session.add(search)
        session.commit()
        session.refresh(search)
        session.refresh(item)

        reference_count = len(existing_images) + saved
        response = SearchCreateResponse(
            search_id=search.id,
            item_id=item.id,
            target_text=search.target_text,
            reference_count=reference_count,
            status=search.status.value,
        )

    return response
