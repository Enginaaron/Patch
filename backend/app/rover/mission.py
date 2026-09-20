"""Per-mission state shared by the controller and the motion layer.

A ``Mission`` is one run of the mission thread (a scan, or an approach). It
captures the controller's generation number at creation; the controller bumps
that number on every start / stop / cancel / replace / override / shutdown, so
"is this mission still allowed to do anything?" is a single comparison made
under the controller lock at the moment of every side effect.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from app.rover.types import Box, MovementPhase, RejectedHint, StopReason

SCAN = "scan"
APPROACH = "approach"


class MissionEnd(Exception):
    """Raised inside a mission thread to finish in a resting phase.

    Every resting outcome (arrived, target_lost, exhausted, stopped, error)
    funnels through this one exception so there is exactly one place that
    stops the wheels, clears the tracking box, persists and publishes.
    """

    def __init__(self, phase: MovementPhase, reason: StopReason, message: str) -> None:
        super().__init__(f"{phase.value}/{reason.value}: {message}")
        self.phase = phase
        self.reason = reason
        self.message = message


@dataclass
class AcceptedTarget:
    """The user-accepted candidate an approach mission drives toward."""

    candidate_id: str
    description: str
    box: Box | None                 # None -> the candidate had no location: never move
    crop_path: str | None = None
    crop_jpeg: bytes | None = None  # loaded on the mission thread, only when box is not None


@dataclass
class Mission:
    search_id: str
    generation: int
    kind: str                       # SCAN | APPROACH
    target_text: str                            # exactly what the user asked for; what OMNI receives
    accepted: AcceptedTarget | None = None      # APPROACH missions started by resume()
    phrase: str = "your item"                   # how the rover says the target back ("your bottle")

    cancel_event: threading.Event = field(default_factory=threading.Event)
    # Set by accept/reject (and by invalidation, purely to wake the waiter).
    decision_event: threading.Event = field(default_factory=threading.Event)
    decision: tuple[str, str] | None = None     # ("accepted" | "rejected", candidate_id)
    pending_candidate_id: str | None = None
    finished: bool = False                      # reached a resting phase on its own
    thread: threading.Thread | None = None

    # Motion / frame bookkeeping. The mission starts from a commanded stop, so
    # "the wheels stopped" at creation time; the first frame it may use must be
    # captured after that (+ settle).
    last_motion_stopped_at: float = field(default_factory=time.monotonic)
    last_used_seq: int = -1
    frame_failures: int = 0
    vision_errors: int = 0

    # Dead-reckoned heading in degrees, clockwise positive. APPROXIMATE: it is
    # turn_degrees_per_second x pulse time, with no encoders behind it.
    heading_est: float = 0.0
    translated: bool = False            # a forward pulse happened (bearings no longer comparable)
    moved_since_candidate: bool = True  # any pulse since the accepted candidate's frame

    scan_steps: int = 0
    approach_pulses: int = 0
    candidates_presented: int = 0
    rejected: list[RejectedHint] = field(default_factory=list)
    rejected_bearings: list[float] = field(default_factory=list)
