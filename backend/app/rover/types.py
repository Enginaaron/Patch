"""Shared types for the rover autonomy layer.

Everything the controller, its ports (drive / camera / vision), the simulation
and the API agree on lives here, so each piece can be built and tested against
the same definitions.

Box convention (unchanged from omni_vision / yolo_detector / bbox):
``[ymin, xmin, ymax, xmax]`` integers on a 0-1000 scale, origin top-left.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

Box = list[int]  # [ymin, xmin, ymax, xmax], 0-1000


class MovementPhase(str, Enum):
    """What the rover body is doing. Deliberately separate from the database
    ``SearchStatus`` (the recognition lifecycle) -- see docs/AUTONOMY.md."""

    IDLE = "idle"
    SCANNING = "scanning"                                  # rotating in bounded steps, looking
    CHECKING = "checking"                                  # wheels stopped, a frame is being inspected
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"  # candidate shown, wheels stopped
    CENTERING = "centering"                                # turning in short pulses to centre the target
    APPROACHING = "approaching"                            # advancing in short pulses
    ARRIVED = "arrived"                                    # conservative proximity confirmed
    TARGET_LOST = "target_lost"                            # accepted target not visible / identity uncertain
    STOPPED = "stopped"                                    # halted by user, override, budget, shutdown...
    EXHAUSTED = "exhausted"                                # scan / candidate budget used up
    ERROR = "error"                                        # camera, inference or internal failure


# A mission thread is alive and owns the wheels in these phases.
ACTIVE_PHASES = frozenset(
    {
        MovementPhase.SCANNING,
        MovementPhase.CHECKING,
        MovementPhase.WAITING_FOR_CONFIRMATION,
        MovementPhase.CENTERING,
        MovementPhase.APPROACHING,
    }
)
# Motor pulses may only be issued in these phases.
MOVING_PHASES = frozenset({MovementPhase.SCANNING, MovementPhase.CENTERING, MovementPhase.APPROACHING})


class StopReason(str, Enum):
    """Why a mission ended up in a resting phase. Stored as plain strings."""

    # ARRIVED
    PROXIMITY_CONFIRMED = "proximity_confirmed"
    # TARGET_LOST
    NOT_VISIBLE = "not_visible"
    IDENTITY_UNCERTAIN = "identity_uncertain"
    NO_BOX = "no_box"                    # found=true but no location -> never move toward it
    # EXHAUSTED
    SCAN_BUDGET = "scan_budget"
    CANDIDATE_BUDGET = "candidate_budget"
    # STOPPED
    USER_STOP = "user_stop"
    CANCELLED = "cancelled"
    MANUAL_OVERRIDE = "manual_override"
    REPLACED = "replaced"
    SHUTDOWN = "shutdown"
    APPROACH_BUDGET = "approach_budget"
    NO_PROGRESS = "no_progress"
    INTERRUPTED = "interrupted"          # process restarted while a mission was active
    # ERROR
    STALE_FRAMES = "stale_frames"
    INFERENCE_FAILED = "inference_failed"
    INVALID_DETECTION = "invalid_detection"
    UNSAFE_VISION = "unsafe_vision"      # simulated vision paired with a real motor driver
    INTERNAL_ERROR = "internal_error"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MovementState:
    """Snapshot of the rover's movement state, as exposed by the API."""

    search_id: str | None = None
    phase: MovementPhase = MovementPhase.IDLE
    message: str = ""
    reason: str | None = None
    active: bool = False                 # a mission thread currently owns the wheels
    seq: int = 0                         # bumps on every published change
    updated_at: datetime = field(default_factory=utcnow)
    scan_steps: int = 0
    scan_budget: int = 0
    approach_pulses: int = 0
    candidates_presented: int = 0
    target_offset: float | None = None   # box centre x - 0.5 (negative = target is left of centre)
    target_width: float | None = None    # box width as a fraction of the frame
    category: str | None = None          # resolved COCO class used for local screening, if any
    vision_mode: str = "live"            # "live" | "simulation"
    driver: str = "sim"                  # "sim" | "gpio"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["phase"] = self.phase.value
        data["updated_at"] = self.updated_at.isoformat()
        return data


# --- frames ---------------------------------------------------------------


@dataclass(frozen=True)
class FramePacket:
    """A captured frame plus the metadata needed to prove it is fresh."""

    frame: np.ndarray
    seq: int               # monotonically increasing capture counter
    captured_at: float     # time.monotonic() when the capture returned
    captured_wall: float   # time.time() at capture, for logs/UI


# --- vision ---------------------------------------------------------------


@dataclass(frozen=True)
class LocalDetection:
    """One local (YOLO) detection of the screened category."""

    box: Box
    confidence: float


@dataclass(frozen=True)
class RejectedHint:
    """An object the user already said is NOT the target."""

    description: str
    crop_jpeg: bytes | None = None


@dataclass(frozen=True)
class IdentifyRequest:
    target_text: str
    reference_images: list[bytes] = field(default_factory=list)
    confirmed_crop: bytes | None = None          # crop of the user-accepted candidate
    rejected: list[RejectedHint] = field(default_factory=list)


@dataclass(frozen=True)
class IdentifyResult:
    """Outcome of one OMNI identification call (single strongest candidate --
    the current schema has no list of alternatives)."""

    found: bool
    likelihood: str                 # "low" | "medium" | "high"
    box: Box | None
    description: str
    latency_seconds: float
    frame_seq: int
    resized_frame: np.ndarray | None = None   # the exact pixels the box refers to
    scene_jpeg: bytes | None = None


@runtime_checkable
class VisionPort(Protocol):
    """Recognition used by the controller. The live implementation wraps
    Aleesha's omni_vision + yolo_detector; simulations set is_simulated=True
    and are refused whenever the motor driver is not the sim driver."""

    is_simulated: bool

    def screen(self, packet: FramePacket, coco_class: str) -> list[LocalDetection] | None:
        """Cheap local screening. Returns every detection of ``coco_class``
        (possibly empty), or None when local screening is unavailable."""
        ...

    def identify(self, packet: FramePacket, request: IdentifyRequest) -> IdentifyResult:
        """Blocking identification call. May raise; the controller bounds it
        with a timeout and discards late results."""
        ...


@runtime_checkable
class CameraPort(Protocol):
    def latest_packet(self, max_age_seconds: float | None = None) -> FramePacket | None: ...

    def wait_for_frame(
        self,
        after_seq: int,
        not_before: float,
        timeout: float,
        cancel: threading.Event | None = None,
    ) -> FramePacket | None:
        """Block until a frame with seq > after_seq and captured_at >= not_before
        (monotonic) exists. None on timeout or when ``cancel`` is set."""
        ...


@runtime_checkable
class DrivePort(Protocol):
    @property
    def driver_name(self) -> str: ...

    def drive(self, command: str, speed: float | None = None, ttl: float | None = None) -> None:
        """Start a named drive command. ``ttl`` arms an independent watchdog
        that stops the wheels even if the caller never returns."""
        ...

    def stop(self) -> None: ...


# --- errors ---------------------------------------------------------------


class RoverError(Exception):
    """Base class for controller errors surfaced to the API."""


class RoverBusyError(RoverError):
    def __init__(self, active_search_id: str, active_target_text: str = "") -> None:
        super().__init__(f"rover is busy with search {active_search_id}")
        self.active_search_id = active_search_id
        self.active_target_text = active_target_text


class SearchNotFoundError(RoverError):
    pass


class InvalidTransitionError(RoverError):
    """The requested change does not apply to the search/candidate's current state."""


class MissionCancelled(Exception):
    """Raised inside a mission thread once its generation token is no longer current."""
