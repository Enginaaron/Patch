"""Database access for the rover controller.

Every status change here is a compare-and-set inside ONE session: the row is
re-read, the expected state is checked, and only then is anything written.
The controller calls these while holding its lock and after re-checking the
mission's generation, so a late result from a cancelled or replaced mission
can never create a candidate or flip a CANCELLED search back to
CANDIDATE_PENDING.

``Search.status`` is the recognition lifecycle; the rover's body lives in the
separate ``RoverMovement`` table (see docs/AUTONOMY.md for the mapping).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import select

from app.config import settings
from app.db import get_session
from app.models import (
    Candidate,
    CandidateDecision,
    Find,
    ReferenceImage,
    RoverMovement,
    Search,
    SearchStatus,
)
from app.rover.geometry import is_valid_box
from app.rover.types import Box, InvalidTransitionError, SearchNotFoundError

logger = logging.getLogger(__name__)


def _naive_utcnow() -> datetime:
    # The existing tables store naive UTC (datetime.utcnow); stay consistent
    # without calling the deprecated function.
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class SearchRecord:
    search_id: str
    item_id: str
    target_text: str
    status: SearchStatus
    reference_paths: tuple[str, ...]


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    search_id: str
    description: str
    box: Box | None
    full_image_path: str
    crop_path: str
    decision: CandidateDecision


@dataclass(frozen=True)
class MovementRecord:
    search_id: str
    phase: str
    message: str
    reason: str | None
    updated_at: datetime    # timezone-aware UTC


def _parse_box(bbox_json: str) -> Box | None:
    try:
        box = json.loads(bbox_json)
    except (TypeError, ValueError):
        return None
    return list(box) if is_valid_box(box) else None


def _candidate_record(candidate: Candidate) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate.id,
        search_id=candidate.search_id,
        description=candidate.description,
        box=_parse_box(candidate.bbox_json),
        full_image_path=candidate.full_image_path,
        crop_path=candidate.crop_path,
        decision=candidate.decision,
    )


class RoverStore:
    """Thin, injectable wrapper over ``app.db.get_session``."""

    # --- reads --------------------------------------------------------------

    def load_search(self, search_id: str) -> SearchRecord | None:
        with get_session() as session:
            search = session.get(Search, search_id)
            if search is None:
                return None
            references = session.exec(
                select(ReferenceImage).where(ReferenceImage.item_id == search.item_id)
            ).all()
            return SearchRecord(
                search_id=search.id,
                item_id=search.item_id,
                target_text=search.target_text,
                status=search.status,
                reference_paths=tuple(r.image_path for r in references),
            )

    def search_status(self, search_id: str) -> SearchStatus | None:
        with get_session() as session:
            search = session.get(Search, search_id)
            return None if search is None else search.status

    def accepted_candidate(self, search_id: str) -> CandidateRecord | None:
        with get_session() as session:
            candidate = session.exec(
                select(Candidate)
                .where(Candidate.search_id == search_id, Candidate.decision == CandidateDecision.ACCEPTED)
                .order_by(Candidate.created_at.desc())
            ).first()
            return None if candidate is None else _candidate_record(candidate)

    def rejected_candidates(self, search_id: str) -> list[CandidateRecord]:
        """Oldest first, so the most recent rejection is last."""
        with get_session() as session:
            rows = session.exec(
                select(Candidate)
                .where(Candidate.search_id == search_id, Candidate.decision == CandidateDecision.REJECTED)
                .order_by(Candidate.created_at)
            ).all()
            return [_candidate_record(row) for row in rows]

    def candidate_count(self, search_id: str) -> int:
        with get_session() as session:
            return len(session.exec(select(Candidate.id).where(Candidate.search_id == search_id)).all())

    def load_movement(self, search_id: str) -> MovementRecord | None:
        with get_session() as session:
            row = session.get(RoverMovement, search_id)
            if row is None:
                return None
            return MovementRecord(
                search_id=row.search_id,
                phase=row.phase,
                message=row.message,
                reason=row.reason,
                updated_at=row.updated_at.replace(tzinfo=timezone.utc),
            )

    # --- compare-and-set writes ----------------------------------------------

    def create_candidate(
        self,
        search_id: str,
        *,
        full_image_path: str,
        crop_path: str,
        box: Box | None,
        description: str,
    ) -> str | None:
        """SEARCHING -> CANDIDATE_PENDING plus a PENDING candidate row.
        Returns the candidate id, or None (nothing written) when the search is
        missing or no longer SEARCHING."""
        with get_session() as session:
            search = session.get(Search, search_id)
            if search is None or search.status != SearchStatus.SEARCHING:
                return None
            candidate = Candidate(
                search_id=search_id,
                full_image_path=full_image_path,
                crop_path=crop_path,
                bbox_json=json.dumps(box),
                description=description,
                decision=CandidateDecision.PENDING,
            )
            session.add(candidate)
            search.status = SearchStatus.CANDIDATE_PENDING
            session.add(search)
            session.commit()
            # commit() expires ORM attributes; read the id inside the session.
            return candidate.id

    def _load_decidable(self, session, search_id: str, candidate_id: str) -> tuple[Search, Candidate]:
        search = session.get(Search, search_id)
        if search is None:
            raise SearchNotFoundError(f"search {search_id} not found")
        candidate = session.get(Candidate, candidate_id)
        if candidate is None or candidate.search_id != search_id:
            raise SearchNotFoundError(f"candidate {candidate_id} not found for search {search_id}")
        if search.status != SearchStatus.CANDIDATE_PENDING or candidate.decision != CandidateDecision.PENDING:
            raise InvalidTransitionError(
                f"candidate is {candidate.decision.value} and search is {search.status.value}; "
                "only a PENDING candidate of a CANDIDATE_PENDING search can be decided"
            )
        return search, candidate

    def accept_candidate(self, search_id: str, candidate_id: str) -> CandidateRecord:
        """CANDIDATE_PENDING -> FOUND, candidate ACCEPTED, ``Find`` row created.
        FOUND means "the user confirmed the identification" -- it says nothing
        about the rover having reached the item."""
        with get_session() as session:
            search, candidate = self._load_decidable(session, search_id, candidate_id)
            candidate.decision = CandidateDecision.ACCEPTED
            search.status = SearchStatus.FOUND
            search.ended_at = _naive_utcnow()
            session.add(candidate)
            session.add(search)
            session.add(
                Find(
                    item_id=search.item_id,
                    search_id=search.id,
                    full_image_path=candidate.full_image_path,
                    crop_path=candidate.crop_path,
                )
            )
            session.commit()
            session.refresh(candidate)
            return _candidate_record(candidate)

    def reject_candidate(self, search_id: str, candidate_id: str) -> CandidateRecord:
        """CANDIDATE_PENDING -> SEARCHING, candidate REJECTED."""
        with get_session() as session:
            search, candidate = self._load_decidable(session, search_id, candidate_id)
            candidate.decision = CandidateDecision.REJECTED
            search.status = SearchStatus.SEARCHING
            session.add(candidate)
            session.add(search)
            session.commit()
            session.refresh(candidate)
            return _candidate_record(candidate)

    def cancel_search(self, search_id: str) -> bool | None:
        """Any status -> CANCELLED. Returns None for an unknown search, True
        when the row changed, False when it was already CANCELLED."""
        with get_session() as session:
            search = session.get(Search, search_id)
            if search is None:
                return None
            if search.status == SearchStatus.CANCELLED:
                return False
            search.status = SearchStatus.CANCELLED
            search.ended_at = _naive_utcnow()
            session.add(search)
            session.commit()
            return True

    def save_movement(
        self, search_id: str, phase: str, message: str, reason: str | None, updated_at: datetime
    ) -> None:
        stamp = updated_at.astimezone(timezone.utc).replace(tzinfo=None) if updated_at.tzinfo else updated_at
        with get_session() as session:
            row = session.get(RoverMovement, search_id)
            if row is None:
                row = RoverMovement(search_id=search_id)
            row.phase = phase
            row.message = message
            row.reason = reason
            row.updated_at = stamp
            session.add(row)
            session.commit()

    # --- media ----------------------------------------------------------------

    def read_media(self, relative_path: str | None) -> bytes | None:
        """Bytes of a file under MEDIA_ROOT, or None when it is missing."""
        if not relative_path:
            return None
        try:
            return Path(settings.media_root, relative_path).read_bytes()
        except OSError:
            logger.warning("media file %s is missing", relative_path)
            return None

    def delete_media(self, relative_path: str | None) -> None:
        if not relative_path:
            return
        try:
            Path(settings.media_root, relative_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("could not delete media file %s", relative_path)
