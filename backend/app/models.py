from datetime import datetime
from enum import Enum
from uuid import uuid4

from sqlmodel import Field, SQLModel


class SearchStatus(str, Enum):
    SEARCHING = "SEARCHING"
    CANDIDATE_PENDING = "CANDIDATE_PENDING"
    FOUND = "FOUND"
    CANCELLED = "CANCELLED"


class CandidateDecision(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class Item(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ReferenceImage(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    item_id: str = Field(foreign_key="item.id")
    image_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Search(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    item_id: str = Field(foreign_key="item.id")
    target_text: str
    status: SearchStatus = Field(default=SearchStatus.SEARCHING)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None


class Candidate(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    search_id: str = Field(foreign_key="search.id")
    full_image_path: str
    crop_path: str
    bbox_json: str
    description: str
    decision: CandidateDecision = Field(default=CandidateDecision.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Find(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    item_id: str = Field(foreign_key="item.id")
    search_id: str = Field(foreign_key="search.id")
    full_image_path: str
    crop_path: str
    found_at: datetime = Field(default_factory=datetime.utcnow)


class RoverMovement(SQLModel, table=True):
    """Last known movement phase of the rover for a search.

    Kept in its own table so the rover's physical state stays separate from
    Search.status (the recognition lifecycle), and so a page refresh or backend
    restart can still report what happened (e.g. "arrived", "target_lost").
    """

    search_id: str = Field(foreign_key="search.id", primary_key=True)
    phase: str = "idle"
    message: str = ""
    reason: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SearchTarget(SQLModel, table=True):
    """How the comprehension layer understood the request behind a search.

    Written once when the search is created (app/services/comprehension.py ->
    POST /api/searches) and read back by the rover controller, so a mission --
    including one resumed after a restart -- uses the same category/landmarks
    without asking a (possibly slow, non-deterministic) provider again.
    ``Search.target_text`` holds the description OMNI is asked to find;
    ``raw_text`` keeps what the user actually typed or said.
    """

    search_id: str = Field(foreign_key="search.id", primary_key=True)
    raw_text: str
    category: str | None = None          # COCO class for local screening; None -> OMNI sampling
    landmarks_json: str = "[]"           # JSON list of strings; context only, never used for gating
    resolver: str = "rule_based_fallback"
    created_at: datetime = Field(default_factory=datetime.utcnow)
