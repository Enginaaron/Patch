import json
import queue
from datetime import datetime
from typing import Iterator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import Session, select

from app.db import get_session
from app.models import Candidate, CandidateDecision, Item, ReferenceImage, Search, SearchStatus
from app.services.event_bus import event_bus
from app.services.search_worker import start_worker, stop_worker
from app.storage import InvalidImageError, save_reference_image

router = APIRouter(prefix="/api")

MAX_REFERENCE_IMAGES = 3


class SearchCreateResponse(BaseModel):
    search_id: str
    item_id: str
    target_text: str
    reference_count: int
    status: str


class ReferenceImageOut(BaseModel):
    id: str
    url: str


class PendingCandidateOut(BaseModel):
    candidate_id: str
    description: str
    crop_url: str
    created_at: datetime


class SearchDetailResponse(BaseModel):
    search_id: str
    item_id: str
    target_text: str
    status: str
    started_at: datetime
    ended_at: datetime | None
    reference_images: list[ReferenceImageOut]
    pending_candidate: PendingCandidateOut | None


def _build_search_detail(session: Session, search: Search) -> SearchDetailResponse:
    reference_images = session.exec(
        select(ReferenceImage).where(ReferenceImage.item_id == search.item_id)
    ).all()

    pending = session.exec(
        select(Candidate)
        .where(Candidate.search_id == search.id, Candidate.decision == CandidateDecision.PENDING)
        .order_by(Candidate.created_at.desc())
    ).first()

    return SearchDetailResponse(
        search_id=search.id,
        item_id=search.item_id,
        target_text=search.target_text,
        status=search.status.value,
        started_at=search.started_at,
        ended_at=search.ended_at,
        reference_images=[
            ReferenceImageOut(id=r.id, url=f"/media/{r.image_path}") for r in reference_images
        ],
        pending_candidate=(
            PendingCandidateOut(
                candidate_id=pending.id,
                description=pending.description,
                crop_url=f"/media/{pending.crop_path}",
                created_at=pending.created_at,
            )
            if pending
            else None
        ),
    )


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

    start_worker(search.id)
    return response


@router.get("/searches/{search_id}", response_model=SearchDetailResponse)
def get_search(search_id: str) -> SearchDetailResponse:
    with get_session() as session:
        search = session.get(Search, search_id)
        if search is None:
            raise HTTPException(status_code=404, detail="search not found")
        return _build_search_detail(session, search)


@router.post("/searches/{search_id}/cancel", response_model=SearchDetailResponse)
def cancel_search(search_id: str) -> SearchDetailResponse:
    with get_session() as session:
        search = session.get(Search, search_id)
        if search is None:
            raise HTTPException(status_code=404, detail="search not found")

        search.status = SearchStatus.CANCELLED
        search.ended_at = datetime.utcnow()
        session.add(search)
        session.commit()
        session.refresh(search)

        detail = _build_search_detail(session, search)

    stop_worker(search_id)
    event_bus.publish(search_id, {"type": "search_cancelled", "payload": {"status": "CANCELLED"}})
    return detail


def _format_sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


# Terminal event types: once one of these fires, the search will never
# change again, so the stream closes itself instead of holding the
# connection open indefinitely.
_TERMINAL_EVENT_TYPES = {"search_cancelled", "search_found"}


@router.get("/searches/{search_id}/events")
def search_events(search_id: str) -> StreamingResponse:
    with get_session() as session:
        search = session.get(Search, search_id)
        if search is None:
            raise HTTPException(status_code=404, detail="search not found")
        initial_status = search.status.value

    def event_stream() -> Iterator[str]:
        subscription = event_bus.subscribe(search_id)
        try:
            # Send the current state immediately so a client that connects
            # after a status change already happened (or that has no
            # separate initial REST fetch) isn't left waiting for the next
            # change that may never come.
            yield _format_sse({"type": "search_status", "payload": {"status": initial_status}})

            while True:
                try:
                    event = subscription.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"  # SSE comment line; keeps proxies/browsers from timing out
                    continue

                yield _format_sse(event)
                if event.get("type") in _TERMINAL_EVENT_TYPES:
                    break
        finally:
            event_bus.unsubscribe(search_id, subscription)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
