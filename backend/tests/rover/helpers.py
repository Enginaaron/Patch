"""Shared scaffolding for the rover controller tests.

Everything runs on the simulated ports (``app.rover.simulation``) against the
temp SQLite database configured in ``tests/conftest.py``. Waiting is done on
the event bus (blocking queue reads with a timeout), never with fixed sleeps.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from typing import Callable

from sqlmodel import select

from app.db import get_session
from app.models import Candidate, Find, Item, RoverMovement, Search, SearchStatus
from app.rover.config import RoverConfig
from app.rover.controller import RoverController
from app.rover.simulation import RecordingDrive, SimCamera, SimDrive, SimVision, SimWorld
from app.rover.types import Box, IdentifyResult
from app.services.event_bus import event_bus

RESTING_PHASES = {"arrived", "target_lost", "stopped", "exhausted", "error"}


def fast_config(**overrides) -> RoverConfig:
    """Tiny durations so a whole mission takes well under a second. The turn
    rate is scaled up to match: one 0.02 s scan pulse is a 20 degree step."""
    values = dict(
        turn_speed=0.4,
        turn_degrees_per_second=1000.0,
        scan_pulse_seconds=0.02,
        scan_max_steps=20,
        scan_direction="right",
        settle_seconds=0.0,
        frame_timeout_seconds=0.2,
        max_frame_failures=3,
        omni_check_every_steps=3,
        max_candidates=5,
        max_vision_errors=3,
        inference_timeout_seconds=5.0,
        camera_hfov_degrees=62.0,
        reject_bearing_tolerance_degrees=15.0,
        candidate_likelihood_threshold="high",
        turn_pulse_min_seconds=0.002,
        turn_pulse_max_seconds=0.03,
        center_tolerance=0.08,
        forward_speed=0.45,
        forward_pulse_seconds=0.02,
        slow_forward_speed=0.35,
        slow_forward_pulse_seconds=0.01,
        slow_zone_fraction=0.7,
        arrival_confirmations=2,
        approach_min_likelihood="medium",
        approach_max_seconds=30.0,
        approach_max_pulses=60,
        stall_pulses=3,
        min_progress_fraction=0.03,
        reacquire_attempts=3,
        omni_revalidate_every_pulses=3,
        association_max_center_shift=0.30,
        association_max_size_ratio=2.0,
        # Generous: the margin only bounds a runaway pulse, it never slows a test.
        pulse_watchdog_margin_seconds=0.5,
        proximity_profiles_json="",
        local_detection_enabled=True,
    )
    values.update(overrides)
    return RoverConfig(**values)


def make_world(cfg: RoverConfig, *, metres_per_pulse: float | None = None, **kwargs) -> SimWorld:
    """World whose made-up physics match ``cfg`` (0.02 s forward = 0.10 m).
    ``metres_per_pulse`` sets how far one normal forward pulse really moves
    the rover (a slower rover, or a target that is far away relative to it)."""
    kwargs.setdefault("forward_metres_per_second", 5.0)
    world = SimWorld.for_config(cfg, **kwargs)
    if metres_per_pulse is not None:
        world.set_forward_step(metres_per_pulse, cfg.forward_pulse_seconds)
    return world


def add_bottle(world: SimWorld, obj_id: str = "target_bottle", *, bearing: float, distance: float, **fields):
    fields.setdefault("coco_class", "bottle")
    fields.setdefault("description", f"{obj_id} (simulated)")
    fields.setdefault("width", 0.08)
    fields.setdefault("height", 0.25)
    return world.add_at(obj_id, bearing=bearing, distance=distance, **fields)


def bottle_resolver(text: str) -> SimpleNamespace:
    """Stand-in for comprehension.resolve_target, so these tests do not depend
    on another module: 'bottle' anywhere in the text -> COCO class bottle."""
    category = "bottle" if "bottle" in text.lower() else None
    return SimpleNamespace(category=category, landmarks=())


# --- database -----------------------------------------------------------------


def make_search(target_text: str = "my bottle", status: SearchStatus = SearchStatus.SEARCHING) -> str:
    with get_session() as session:
        item = Item(name=target_text)
        session.add(item)
        session.commit()
        search = Search(item_id=item.id, target_text=target_text, status=status)
        session.add(search)
        session.commit()
        return search.id


def search_status(search_id: str) -> SearchStatus:
    with get_session() as session:
        return session.get(Search, search_id).status


def get_search(search_id: str) -> Search:
    with get_session() as session:
        return session.get(Search, search_id)


def candidates_of(search_id: str) -> list[Candidate]:
    with get_session() as session:
        return list(
            session.exec(select(Candidate).where(Candidate.search_id == search_id).order_by(Candidate.created_at)).all()
        )


def finds_of(search_id: str) -> list[Find]:
    with get_session() as session:
        return list(session.exec(select(Find).where(Find.search_id == search_id)).all())


def movement_row(search_id: str) -> RoverMovement | None:
    with get_session() as session:
        return session.get(RoverMovement, search_id)


# --- events ---------------------------------------------------------------------


class EventLog:
    """Subscribes to one search on the event bus. ``wait_*`` consume events in
    order (a cursor), so "the next candidate_found" means what it says."""

    def __init__(self, search_id: str) -> None:
        self.search_id = search_id
        self._queue = event_bus.subscribe(search_id)
        self.events: list[dict] = []
        self.times: list[float] = []
        self._cursor = 0

    def close(self) -> None:
        event_bus.unsubscribe(self.search_id, self._queue)

    def _pull(self, timeout: float) -> bool:
        try:
            event = self._queue.get(timeout=timeout) if timeout > 0 else self._queue.get_nowait()
        except queue.Empty:
            return False
        self.events.append(event)
        self.times.append(time.monotonic())
        return True

    def drain(self) -> list[dict]:
        while self._pull(0):
            pass
        return self.events

    def wait_for(self, predicate: Callable[[dict], bool], timeout: float = 10.0, what: str = "event") -> dict:
        deadline = time.monotonic() + timeout
        while True:
            while self._cursor < len(self.events):
                event = self.events[self._cursor]
                self._cursor += 1
                if predicate(event):
                    return event
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._pull(remaining):
                if time.monotonic() >= deadline:
                    raise AssertionError(f"timed out waiting for {what}; saw {self.summary()}")

    def wait_type(self, event_type: str, timeout: float = 10.0) -> dict:
        return self.wait_for(lambda e: e.get("type") == event_type, timeout, what=event_type)

    def wait_phase(self, *phases: str, timeout: float = 10.0) -> dict:
        wanted = set(phases)
        return self.wait_for(
            lambda e: e.get("type") == "movement" and e["payload"]["phase"] in wanted,
            timeout,
            what=f"phase in {sorted(wanted)}",
        )

    def wait_rest(self, timeout: float = 15.0) -> dict:
        """The next movement event that is a resting phase."""
        return self.wait_phase(*RESTING_PHASES, timeout=timeout)

    def summary(self) -> list[str]:
        out = []
        for event in self.events:
            if event.get("type") == "movement":
                payload = event["payload"]
                out.append(f"movement:{payload['phase']}:{payload['reason']}")
            else:
                out.append(str(event.get("type")))
        return out

    def types(self) -> list[str]:
        return [str(e.get("type")) for e in self.drain()]

    def phases(self) -> list[str]:
        return [e["payload"]["phase"] for e in self.drain() if e.get("type") == "movement"]

    def messages(self) -> list[str]:
        return [e["payload"]["message"] for e in self.drain() if e.get("type") == "movement"]

    def of_type(self, event_type: str) -> list[dict]:
        return [e for e in self.drain() if e.get("type") == event_type]


class TrackingSink:
    """Stands in for yolo_detector.set_tracking_box and remembers every call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.history: list[tuple[Box | None, float]] = []

    def __call__(self, box: Box | None) -> None:
        with self._lock:
            self.history.append((None if box is None else list(box), time.monotonic()))

    @property
    def value(self) -> Box | None:
        with self._lock:
            return self.history[-1][0] if self.history else None


class Harness:
    """A controller wired to simulated ports, with everything observable."""

    def __init__(
        self,
        cfg: RoverConfig | None = None,
        world: SimWorld | None = None,
        *,
        vision=None,
        camera=None,
        driver_name: str | None = None,
        resolver=bottle_resolver,
    ) -> None:
        self.cfg = cfg if cfg is not None else fast_config()
        self.world = world if world is not None else make_world(self.cfg)
        self.drive = RecordingDrive(SimDrive(self.world), driver_name=driver_name)
        self.camera = camera if camera is not None else SimCamera(self.world)
        self.vision = vision if vision is not None else SimVision(self.world)
        self.tracking = TrackingSink()
        self.controller = RoverController(
            self.drive, self.camera, self.vision, self.cfg, resolver=resolver, tracking_sink=self.tracking
        )
        self._logs: list[EventLog] = []
        self._releases: list[Callable[[], None]] = []

    def search(self, target_text: str = "my bottle") -> tuple[str, EventLog]:
        """A SEARCHING search plus an event log subscribed BEFORE anything starts."""
        search_id = make_search(target_text)
        return search_id, self.log(search_id)

    def log(self, search_id: str) -> EventLog:
        log = EventLog(search_id)
        self._logs.append(log)
        return log

    def on_close(self, release: Callable[[], None]) -> None:
        self._releases.append(release)

    def new_controller(self) -> RoverController:
        """A second controller on the same world and database: what a backend
        restart looks like (no live mission, fresh generation counter)."""
        return RoverController(
            self.drive, self.camera, self.vision, self.cfg, resolver=bottle_resolver, tracking_sink=self.tracking
        )

    def close(self) -> None:
        # Unblock anything a test left hanging, THEN shut down (joins threads).
        for release in self._releases:
            release()
        release = getattr(self.vision, "release", None)
        if callable(release):
            release()
        self.controller.shutdown()
        for log in self._logs:
            log.close()


class ScriptedVision:
    """A VisionPort that answers from a function -- for cases the simulated
    world cannot produce (boxes that never grow, flickering sizes, garbage)."""

    def __init__(self, answer: Callable[[int], dict], *, is_simulated: bool = True) -> None:
        self.is_simulated = is_simulated
        self._answer = answer
        self._lock = threading.Lock()
        self.identify_calls = 0
        self.frame_seqs: list[int] = []

    def screen(self, packet, coco_class):
        return None  # no local detector: OMNI on every observation

    def identify(self, packet, request):
        with self._lock:
            self.identify_calls += 1
            index = self.identify_calls
            self.frame_seqs.append(packet.seq)
        spec = dict(found=True, likelihood="high", box=None, description="scripted object")
        spec.update(self._answer(index))
        return IdentifyResult(latency_seconds=0.0, frame_seq=packet.seq, **spec)
