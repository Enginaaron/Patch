"""Search lifecycle API.

These handlers are deliberately thin. Every state transition -- starting a
mission, accepting / rejecting a candidate, resuming, cancelling -- lives in the
rover controller (``app/rover/controller.py``), which is the one owner of the
wheels. The router validates input, creates the rows for a new search, maps
controller errors to HTTP and builds the detail response.

Two separate things are reported for a search (see docs/AUTONOMY.md):

* ``status``   -- the recognition lifecycle stored on ``Search``. FOUND means
  "the user confirmed Patch identified the item". It never means the rover
  reached it.
* ``movement`` -- what the rover body is doing. Arrival is only ever
  ``movement.phase == "arrived"``.
"""

import asyncio
import json
import logging
import queue
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any, AsyncIterator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import Session, select

from app.db import get_session
from app.models import Candidate, CandidateDecision, Item, ReferenceImage, RoverMovement, Search, SearchStatus
from app.rover.controller import get_rover_controller
from app.rover.geometry import is_valid_box
from app.rover.types import (
    InvalidTransitionError,
    MovementPhase,
    MovementState,
    RoverBusyError,
    RoverError,
    SearchNotFoundError,
)
from app.services.comprehension import ResolvedTarget, resolve_target
from app.services.event_bus import event_bus
from app.storage import InvalidImageError, save_reference_image

logger = logging.getLogger(__name__)

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
    image_url: str
    # None: OMNI said "found" without a location. The question may still be
    # asked, but the rover will never drive toward a candidate without a box.
    box: list[int] | None
    created_at: datetime


class AcceptedCandidateOut(BaseModel):
    candidate_id: str
    description: str
    crop_url: str


class ResolvedTargetOut(BaseModel):
    category: str | None
    landmarks: list[str]
    resolver: str


class SearchDetailResponse(BaseModel):
    search_id: str
    item_id: str
    target_text: str
    status: str
    started_at: datetime
    ended_at: datetime | None
    reference_images: list[ReferenceImageOut]
    pending_candidate: PendingCandidateOut | None
    accepted_candidate: AcceptedCandidateOut | None
    resolved_target: ResolvedTargetOut | None
    movement: dict[str, Any]   # MovementState.to_dict()
    can_resume: bool
    terminal: bool             # nothing about this search will change again


# --- helpers -----------------------------------------------------------------


def _is_terminal(status: SearchStatus, movement: MovementState) -> bool:
    """CANCELLED, or physically arrived. FOUND alone is NOT terminal: after the
    user accepts a candidate the rover still has to drive to it."""
    return status == SearchStatus.CANCELLED or movement.phase == MovementPhase.ARRIVED


def _parse_box(bbox_json: str) -> list[int] | None:
    try:
        box = json.loads(bbox_json)
    except (TypeError, ValueError):
        return None
    return list(box) if is_valid_box(box) else None


# Resolved once per search and remembered. A registered comprehension provider
# may be slow (a model call), and the search page refetches the detail on every
# SSE message -- it must not pay for a fresh resolution each time.
_RESOLVED_CACHE_SIZE = 128
_resolved_lock = threading.Lock()
_resolved_cache: "OrderedDict[str, ResolvedTargetOut]" = OrderedDict()


def _remember_resolved(search_id: str, resolved: ResolvedTarget) -> ResolvedTargetOut:
    out = ResolvedTargetOut(
        category=resolved.category, landmarks=list(resolved.landmarks), resolver=resolved.resolver
    )
    with _resolved_lock:
        _resolved_cache[search_id] = out
        _resolved_cache.move_to_end(search_id)
        while len(_resolved_cache) > _RESOLVED_CACHE_SIZE:
            _resolved_cache.popitem(last=False)
    return out


def _resolved_target(search_id: str, target_text: str) -> ResolvedTargetOut | None:
    with _resolved_lock:
        cached = _resolved_cache.get(search_id)
    if cached is not None:
        return cached
    try:
        return _remember_resolved(search_id, resolve_target(target_text))
    except Exception:  # noqa: BLE001 - informational only; never fail the detail over it
        logger.exception("search %s: could not resolve target %r", search_id, target_text)
        return None


def _build_search_detail(session: Session, search: Search) -> SearchDetailResponse:
    reference_images = session.exec(
        select(ReferenceImage).where(ReferenceImage.item_id == search.item_id)
    ).all()

    # "Pending" only means something while the search is waiting for that
    # answer: a CANCELLED search keeps its candidate row PENDING (the decision
    # is left unchanged on cancel), and must not show a question any more.
    pending = None
    if search.status == SearchStatus.CANDIDATE_PENDING:
        pending = session.exec(
            select(Candidate)
            .where(Candidate.search_id == search.id, Candidate.decision == CandidateDecision.PENDING)
            .order_by(Candidate.created_at.desc())
        ).first()

    accepted = session.exec(
        select(Candidate)
        .where(Candidate.search_id == search.id, Candidate.decision == CandidateDecision.ACCEPTED)
        .order_by(Candidate.created_at.desc())
    ).first()

    controller = get_rover_controller()
    movement = controller.state_for(search.id)

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
                image_url=f"/media/{pending.full_image_path}",
                box=_parse_box(pending.bbox_json),
                created_at=pending.created_at,
            )
            if pending
            else None
        ),
        accepted_candidate=(
            AcceptedCandidateOut(
                candidate_id=accepted.id,
                description=accepted.description,
                crop_url=f"/media/{accepted.crop_path}",
            )
            if accepted
            else None
        ),
        resolved_target=_resolved_target(search.id, search.target_text),
        movement=movement.to_dict(),
        can_resume=controller.can_resume(search.id),
        terminal=_is_terminal(search.status, movement),
    )


def _search_detail(search_id: str) -> SearchDetailResponse:
    with get_session() as session:
        search = session.get(Search, search_id)
        if search is None:
            raise HTTPException(status_code=404, detail="search not found")
        return _build_search_detail(session, search)


def _busy_error(active_search_id: str, active_target_text: str) -> HTTPException:
    # Patch has one body: one search owns it at a time. The client may offer
    # to resubmit with replace_active=true.
    return HTTPException(
        status_code=409,
        detail={
            "code": "rover_busy",
            "message": f"Patch is already looking for “{active_target_text}”. Cancel that search or replace it.",
            "active_search_id": active_search_id,
            "active_target_text": active_target_text,
        },
    )


def _http_error(exc: RoverError) -> HTTPException:
    """Controller error -> HTTP. 404 unknown search/candidate, 409 for a change
    that does not apply right now, 503 when the controller is shut down."""
    if isinstance(exc, SearchNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, RoverBusyError):
        return _busy_error(exc.active_search_id, exc.active_target_text)
    if isinstance(exc, InvalidTransitionError):
        return HTTPException(status_code=409, detail={"code": "invalid_transition", "message": str(exc)})
    return HTTPException(status_code=503, detail={"code": "rover_unavailable", "message": str(exc)})


def _discard_search(search_id: str) -> None:
    """Undo the rows of a search whose mission could not be started. The 409 /
    503 contract is "no search was created", also when the rover was claimed
    in the instant between the busy check and ``start_search``."""
    with _resolved_lock:
        _resolved_cache.pop(search_id, None)
    with get_session() as session:
        movement = session.get(RoverMovement, search_id)
        if movement is not None:
            session.delete(movement)
        search = session.get(Search, search_id)
        if search is not None:
            session.delete(search)
        session.commit()


# --- create / read -------------------------------------------------------------


@router.post("/searches", response_model=SearchCreateResponse)
async def create_search(
    target_text: str = Form(...),
    reference_images: list[UploadFile] = File(default=[]),
    replace_active: bool = Form(False),
) -> SearchCreateResponse:
    text = target_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="target_text must not be empty")

    if len(reference_images) > MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=400, detail=f"at most {MAX_REFERENCE_IMAGES} reference images are allowed"
        )

    # Understand the request BEFORE touching anything. "The other one" names
    # nothing the rover could look for: ask, create no search, leave the rover
    # (and whatever search it is on) alone. Off the event loop, because a
    # registered provider may be slow and /api/rover/stop runs on that loop.
    try:
        resolved: ResolvedTarget | None = await run_in_threadpool(resolve_target, text)
    except Exception:  # noqa: BLE001 - comprehension is a helper; the controller falls back to OMNI sampling
        logger.exception("could not resolve target %r; searching with the text as given", text)
        resolved = None
    if resolved is not None and resolved.needs_clarification:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "needs_clarification",
                "question": resolved.clarification_question,
                "raw_text": text,
            },
        )

    controller = get_rover_controller()
    if not replace_active:
        active = controller.snapshot()
        if active["active_search_id"] is not None:
            raise _busy_error(active["active_search_id"], active["active_target_text"] or "")

    normalized = text.lower()

    # no_autoflush: nothing is written before the commit below. With autoflush
    # the first query after session.add(item) INSERTs it, and that SQLite write
    # transaction would then stay open across the `await`s that read the
    # uploads. A stop served by the event loop in the meantime (its movement
    # write needs the database) would block the loop on that lock, this handler
    # could never get back to commit, and every request would hang until
    # SQLite's busy timeout.
    with get_session() as session, session.no_autoflush:
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

    if resolved is not None:
        _remember_resolved(response.search_id, resolved)
    try:
        # With replace_active the controller cancels the old search exactly
        # like a user cancel (halts first, CANCELLED, search_cancelled event
        # with reason "replaced") and only then starts this one.
        controller.start_search(response.search_id, replace=replace_active)
    except RoverError as exc:
        _discard_search(response.search_id)
        raise _http_error(exc) from exc
    return response


@router.get("/searches/{search_id}", response_model=SearchDetailResponse)
def get_search(search_id: str) -> SearchDetailResponse:
    return _search_detail(search_id)


# --- candidate decisions / resume / cancel ---------------------------------------


@router.post("/searches/{search_id}/candidates/{candidate_id}/accept", response_model=SearchDetailResponse)
def accept_candidate(search_id: str, candidate_id: str) -> SearchDetailResponse:
    """The user said "yes, that's it": CANDIDATE_PENDING -> FOUND, and the rover
    starts to approach. FOUND is the identification being confirmed, not an
    arrival."""
    try:
        get_rover_controller().accept_candidate(search_id, candidate_id)
    except RoverError as exc:
        raise _http_error(exc) from exc
    return _search_detail(search_id)


@router.post("/searches/{search_id}/candidates/{candidate_id}/reject", response_model=SearchDetailResponse)
def reject_candidate(search_id: str, candidate_id: str) -> SearchDetailResponse:
    try:
        get_rover_controller().reject_candidate(search_id, candidate_id)
    except RoverError as exc:
        raise _http_error(exc) from exc
    return _search_detail(search_id)


@router.post("/searches/{search_id}/resume", response_model=SearchDetailResponse)
def resume_search(search_id: str) -> SearchDetailResponse:
    """Start moving again after a stop, a manual override, a lost target or a
    backend restart. Nothing resumes by itself -- only through this call."""
    try:
        get_rover_controller().resume(search_id)
    except RoverError as exc:
        raise _http_error(exc) from exc
    return _search_detail(search_id)


@router.post("/searches/{search_id}/cancel", response_model=SearchDetailResponse)
async def cancel_search(search_id: str) -> SearchDetailResponse:
    # Inline on the event loop, like /api/rover/stop: cancelling halts the
    # rover FIRST (then marks the search CANCELLED), and that halt must not
    # queue behind a busy threadpool. The controller call never blocks; only
    # the detail read afterwards goes to the threadpool. Idempotent.
    if not get_rover_controller().cancel_search(search_id):
        raise HTTPException(status_code=404, detail="search not found")
    return await run_in_threadpool(_search_detail, search_id)


# --- server-sent events ------------------------------------------------------------

_KEEPALIVE_SECONDS = 15.0
# The event bus hands events over on plain thread-safe queues (they are
# published from the mission thread). The stream polls its queue from the event
# loop instead of blocking a threadpool thread per open page, so any number of
# open search pages cannot starve the accept / reject / detail handlers.
_POLL_SECONDS = 0.05


def _format_sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def _status_snapshot(search_id: str) -> dict | None:
    with get_session() as session:
        search = session.get(Search, search_id)
        if search is None:
            return None
        status = search.status
    movement = get_rover_controller().state_for(search_id)
    terminal = _is_terminal(status, movement)
    return {
        "type": "search_status",
        "payload": {"status": status.value, "movement": movement.to_dict(), "terminal": terminal},
        "terminal": terminal,
    }


async def _event_stream(
    search_id: str,
    *,
    keepalive_seconds: float = _KEEPALIVE_SECONDS,
    poll_seconds: float = _POLL_SECONDS,
) -> AsyncIterator[str]:
    """Messages are refetch signals: clients re-read GET /searches/{id} on each
    one and close on ``terminal: true``."""
    # Subscribe BEFORE reading the snapshot. Anything that changes while the
    # snapshot is being read is then already queued, so there is no gap in
    # which an event could be missed (a duplicate refetch is harmless).
    subscription = event_bus.subscribe(search_id)
    try:
        # The current state goes out immediately, so a client that connects (or
        # auto-reconnects) after a change is not left waiting for the next one.
        first = await run_in_threadpool(_status_snapshot, search_id)
        if first is None:
            return
        yield _format_sse(first)
        if first["terminal"]:
            # Nothing will ever change again: close instead of holding the
            # connection open (the client closes too, so it does not reconnect).
            return

        last_sent = time.monotonic()
        while True:
            try:
                event = subscription.get_nowait()
            except queue.Empty:
                if time.monotonic() - last_sent >= keepalive_seconds:
                    yield ": keep-alive\n\n"  # SSE comment line; keeps proxies/browsers from timing out
                    last_sent = time.monotonic()
                await asyncio.sleep(poll_seconds)
                continue

            yield _format_sse(event)
            last_sent = time.monotonic()
            if event.get("terminal") is True:
                return
    finally:
        event_bus.unsubscribe(search_id, subscription)


@router.get("/searches/{search_id}/events")
def search_events(search_id: str) -> StreamingResponse:
    with get_session() as session:
        if session.get(Search, search_id) is None:
            raise HTTPException(status_code=404, detail="search not found")

    return StreamingResponse(
        _event_stream(search_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
