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
