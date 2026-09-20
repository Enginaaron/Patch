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
from app.models import Candidate, CandidateDecision, Find, Item, ReferenceImage, Search, SearchStatus
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
    image_url: str
    crop_url: str
    box: list[int] | None
    created_at: datetime


class FindOut(BaseModel):
    image_url: str
    crop_url: str
    found_at: datetime


class SearchDetailResponse(BaseModel):
    search_id: str
    item_id: str
    target_text: str
    status: str
    started_at: datetime
    ended_at: datetime | None
    reference_images: list[ReferenceImageOut]
    pending_candidate: PendingCandidateOut | None
    find: FindOut | None


def _build_search_detail(session: Session, search: Search) -> SearchDetailResponse:
    reference_images = session.exec(
        select(ReferenceImage).where(ReferenceImage.item_id == search.item_id)
    ).all()

    pending = session.exec(
        select(Candidate)
        .where(Candidate.search_id == search.id, Candidate.decision == CandidateDecision.PENDING)
        .order_by(Candidate.created_at.desc())
    ).first()

    find = session.exec(
        select(Find).where(Find.search_id == search.id).order_by(Find.found_at.desc())
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
                image_url=f"/media/{pending.full_image_path}",
                crop_url=f"/media/{pending.crop_path}",
                box=json.loads(pending.bbox_json) if pending.bbox_json else None,
                created_at=pending.created_at,
            )
            if pending
            else None
        ),
        find=(
            FindOut(
                image_url=f"/media/{find.full_image_path}",
                crop_url=f"/media/{find.crop_path}",
                found_at=find.found_at,
            )
            if find
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


def _get_pending_candidate(session: Session, candidate_id: str) -> Candidate:
    candidate = session.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    if candidate.decision != CandidateDecision.PENDING:
        raise HTTPException(status_code=409, detail="candidate has already been decided")
    return candidate


# Spec 15 gives an explicit, search-id-free contract for these routes
# (POST /api/candidates/{candidate_id}/reject). candidate_id is already
# globally unique, so confirm was moved to match rather than leaving one
# endpoint nested under /searches/{search_id} and the other flat.
@router.post("/candidates/{candidate_id}/confirm", response_model=SearchDetailResponse)
def confirm_candidate(candidate_id: str) -> SearchDetailResponse:
    """Spec 14/16: 'THAT'S IT' -- identical whether triggered by tap or by the
    voice keyword match, since both call this same endpoint."""
    with get_session() as session:
        candidate = _get_pending_candidate(session, candidate_id)

        search = session.get(Search, candidate.search_id)
        if search is None:
            raise HTTPException(status_code=404, detail="search not found")

        candidate.decision = CandidateDecision.ACCEPTED
        session.add(candidate)

        find = Find(
            item_id=search.item_id,
            search_id=search.id,
            full_image_path=candidate.full_image_path,
            crop_path=candidate.crop_path,
        )
        session.add(find)

        search.status = SearchStatus.FOUND
        search.ended_at = datetime.utcnow()
        session.add(search)
        session.commit()
        session.refresh(search)

        search_id = search.id
        detail = _build_search_detail(session, search)

    # The worker already stopped itself when this candidate was created
    # (Spec 12), but call it explicitly too -- FOUND is terminal, and this
    # makes "stop the worker" (Spec 16) true regardless of that timing,
    # rather than relying on an implicit side effect. A no-op if it already
    # unregistered itself.
    stop_worker(search_id)
    event_bus.publish(search_id, {"type": "search_found", "payload": {"status": "FOUND"}})
    return detail


@router.post("/candidates/{candidate_id}/reject", response_model=SearchDetailResponse)
def reject_candidate(candidate_id: str) -> SearchDetailResponse:
    """Spec 15: 'NOT MINE' -- identical whether triggered by tap or by the
    voice keyword match, since both call this same endpoint. The rejected
    candidate's row (and its crop image on disk) is left untouched -- only
    its decision flips to REJECTED -- so search_worker._load_recent_rejected_crops
    can feed it back into subsequent OMNI prompts."""
    with get_session() as session:
        candidate = _get_pending_candidate(session, candidate_id)

        search = session.get(Search, candidate.search_id)
        if search is None:
            raise HTTPException(status_code=404, detail="search not found")

        candidate.decision = CandidateDecision.REJECTED
        session.add(candidate)

        search.status = SearchStatus.SEARCHING
        session.add(search)
        session.commit()
        session.refresh(search)

        search_id = search.id
        detail = _build_search_detail(session, search)

    event_bus.publish(search_id, {"type": "candidate_rejected", "payload": {"candidate_id": candidate_id}})
    start_worker(search_id)  # resume polling -- the worker stopped itself when this candidate fired
    event_bus.publish(search_id, {"type": "search_resumed", "payload": {"status": "SEARCHING"}})
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
