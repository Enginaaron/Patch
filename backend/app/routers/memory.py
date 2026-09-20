import logging
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import select

from app.config import settings
from app.db import get_session
from app.models import Find, Item, ReferenceImage, Search, SearchStatus
from app.routers.searches import SearchCreateResponse, _discard_search, _http_error
from app.rover.types import RoverError
from app.services.camera_service import camera_service
from app.services.omni_vision import detect_personalized
from app.services.search_worker import FRAME_TARGET_WIDTH, JPEG_QUALITY, start_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class FindListItem(BaseModel):
    find_id: str
    item_id: str
    item_name: str
    image_url: str
    crop_url: str
    found_at: datetime


class ItemReferenceImageOut(BaseModel):
    id: str
    url: str


class ItemFindOut(BaseModel):
    find_id: str
    image_url: str
    crop_url: str
    found_at: datetime


class ItemDetailResponse(BaseModel):
    item_id: str
    name: str
    reference_images: list[ItemReferenceImageOut]
    finds: list[ItemFindOut]


class QuickCheckResponse(BaseModel):
    visible: bool
    checked_at: datetime


def _encode_frame(frame) -> bytes:
    """Same resize/quality as search_worker._prepare_frame, duplicated
    locally since Quick Check doesn't need the resized array back (no crop
    math -- box_2d is ignored entirely here, per Spec 19)."""
    height, width = frame.shape[:2]
    scale = FRAME_TARGET_WIDTH / width
    resized = cv2.resize(frame, (FRAME_TARGET_WIDTH, round(height * scale)))
    ok, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("failed to JPEG-encode frame")
    return buffer.tobytes()


@router.get("/finds", response_model=list[FindListItem])
def list_finds() -> list[FindListItem]:
    with get_session() as session:
        rows = session.exec(
            select(Find, Item).join(Item, Find.item_id == Item.id).order_by(Find.found_at.desc())
        ).all()

        return [
            FindListItem(
                find_id=find.id,
                item_id=item.id,
                item_name=item.name,
                image_url=f"/media/{find.full_image_path}",
                crop_url=f"/media/{find.crop_path}",
                found_at=find.found_at,
            )
            for find, item in rows
        ]


@router.get("/items/{item_id}", response_model=ItemDetailResponse)
def get_item(item_id: str) -> ItemDetailResponse:
    with get_session() as session:
        item = session.get(Item, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="item not found")

        reference_images = session.exec(
            select(ReferenceImage).where(ReferenceImage.item_id == item_id)
        ).all()

        finds = session.exec(
            select(Find).where(Find.item_id == item_id).order_by(Find.found_at.desc())
        ).all()

        return ItemDetailResponse(
            item_id=item.id,
            name=item.name,
            reference_images=[
                ItemReferenceImageOut(id=r.id, url=f"/media/{r.image_path}") for r in reference_images
            ],
            finds=[
                ItemFindOut(
                    find_id=f.id,
                    image_url=f"/media/{f.full_image_path}",
                    crop_url=f"/media/{f.crop_path}",
                    found_at=f.found_at,
                )
                for f in finds
            ],
        )


@router.post("/items/{item_id}/search", response_model=SearchCreateResponse)
def find_again(item_id: str) -> SearchCreateResponse:
    """Spec 18: reuse an existing Item -- same target text, same reference
    images (ReferenceImage rows are already keyed by item_id, not search_id,
    so a new Search against the same item_id picks them up automatically;
    nothing is re-uploaded or duplicated on disk)."""
    with get_session() as session:
        item = session.get(Item, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="item not found")

        reference_count = len(
            session.exec(select(ReferenceImage).where(ReferenceImage.item_id == item_id)).all()
        )

        search = Search(item_id=item.id, target_text=item.name, status=SearchStatus.SEARCHING)
        session.add(search)
        session.commit()
        session.refresh(search)

        response = SearchCreateResponse(
            search_id=search.id,
            item_id=item.id,
            target_text=search.target_text,
            reference_count=reference_count,
            status=search.status.value,
        )

    try:
        start_worker(response.search_id)
    except RoverError as exc:
        _discard_search(response.search_id)
        raise _http_error(exc) from exc
    return response


@router.post("/items/{item_id}/quick-check", response_model=QuickCheckResponse)
def quick_check(item_id: str) -> QuickCheckResponse:
    """Spec 19: a stateless, one-shot "is it still there" check -- a single
    Mode B OMNI call against the current frame, no Search/Candidate row and
    no event_bus activity. box_2d is deliberately ignored; only `found`
    is surfaced, mapped to `visible`."""
    with get_session() as session:
        item = session.get(Item, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="item not found")

        reference_images = session.exec(
            select(ReferenceImage).where(ReferenceImage.item_id == item_id)
        ).all()
        if not reference_images:
            raise HTTPException(
                status_code=400,
                detail="This item has no reference photos yet -- teach it first before running a quick check.",
            )
        reference_paths = [r.image_path for r in reference_images]
        target_text = item.name

    frame = camera_service.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera is unavailable right now.")

    scene_jpeg = _encode_frame(frame)
    reference_bytes = [Path(settings.media_root, p).read_bytes() for p in reference_paths]

    try:
        result = detect_personalized(target_text, reference_bytes, scene_jpeg)
    except Exception as exc:
        logger.exception("quick check OMNI call failed for item %s", item_id)
        raise HTTPException(status_code=502, detail=f"vision check failed: {exc}") from exc

    return QuickCheckResponse(visible=result.detection.found, checked_at=datetime.utcnow())
