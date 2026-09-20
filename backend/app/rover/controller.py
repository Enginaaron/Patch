"""The one controller that owns the wheels during an autonomous search.

Design in one paragraph: a search is driven by a single mission thread that
alternates "look while stopped" and "move in one short pulse". Every mission
captures a *generation* number. Any start / stop / cancel / replace / manual
override / shutdown bumps the controller's generation under ``_lock`` and stops
the wheels before returning; from that instant the old mission fails the
currency check that guards EVERY side effect it could still attempt (motor
command, candidate row, status change, RoverMovement write, event, tracking
box). A late OMNI result from a cancelled mission therefore cannot move the
rover, create a candidate, or flip a CANCELLED search back to
CANDIDATE_PENDING -- and an old thread's cleanup cannot disturb its
replacement.

Lock discipline: ``_lock`` is only ever held for short, non-blocking work
(state, SQLite compare-and-set, event hand-off, a motor command). It is never
held during inference, frame waits, ``Event.wait``, sleeps or joins, and never
while the wheels are turning -- so ``stop()`` is never queued behind anything
slow and can safely run inline on the event loop.

``Search.status`` (recognition lifecycle) and ``MovementPhase`` (the body) are
separate on purpose: FOUND means the user confirmed the identification, never
that the rover reached the item. Arrival is only ``phase == arrived``.

Honest limits: headings are dead-reckoned from pulse time (no encoders), box
sizes are not distances, YOLO does not track identity, and nothing here has
been verified on the physical rover yet.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable

from app.config import settings
from app.models import SearchStatus
from app.rover import geometry
from app.rover.config import RoverConfig
from app.rover.mission import APPROACH, SCAN, AcceptedTarget, Mission, MissionEnd
from app.rover.motion import MAX_PULSE_SECONDS, UNSAFE_VISION_MESSAGE, Motion, vision_is_unsafe
from app.rover.proximity import ProximityProfile, in_slow_zone, load_profiles, profile_for
from app.rover.steering import ApproachSteering
from app.rover.store import MovementRecord, RoverStore, SearchRecord
from app.rover.types import (
    ACTIVE_PHASES,
    Box,
    CameraPort,
    DrivePort,
    FramePacket,
    IdentifyRequest,
    IdentifyResult,
    InvalidTransitionError,
    LocalDetection,
    MissionCancelled,
    MovementPhase,
    MovementState,
    RejectedHint,
    RoverBusyError,
    RoverError,
    SearchNotFoundError,
    StopReason,
    VisionPort,
    utcnow,
)
from app.services import search_worker
from app.services.event_bus import event_bus

logger = logging.getLogger(__name__)

_INFER_SLICE_SECONDS = 0.05      # how often a waiting inference re-checks cancellation
_DECISION_SLICE_SECONDS = 1.0    # safety net only: accept/reject/invalidate all set the event
_JOIN_BUDGET_SECONDS = 3.0       # shutdown(): total time spent joining mission threads
_LIKELIHOODS = ("low", "medium", "high")

_UNREAD: Any = object()          # sentinel: "the RoverMovement row has not been read yet"

_INTERNAL_ERROR_MESSAGE = "Something went wrong — I've stopped"
_INTERRUPTED_MESSAGE = "I was interrupted — press resume to continue"


@dataclass
class _Context:
    """Resolved once per mission, on the mission thread."""

    category: str | None
    landmarks: tuple[str, ...]
    references: list[bytes]
    profile: ProximityProfile


@dataclass
class _Track:
    """The accepted target as last observed. ``source`` says who vouched for
    the box: "omni" (identity validated) or "yolo" (category box associated
    by position/size only)."""

    box: Box
    source: str
    pulses_since_omni: int = 0
    confirmed_crop: bytes | None = None


class RoverController:
    def __init__(
        self,
        drive: DrivePort,
        camera: CameraPort,
        vision: VisionPort,
        config: RoverConfig,
        events: Any = event_bus,
        store: RoverStore | None = None,
        *,
        resolver: Callable[[str], Any] | None = None,
        tracking_sink: Callable[[Box | None], None] | None = None,
    ) -> None:
        self._drive = drive
        self._camera = camera
        self._vision = vision
        self._config = config
        self._events = events
        self._store = store if store is not None else RoverStore()
        self._resolver = resolver
        if tracking_sink is None:
            # The MJPEG stream draws whatever box is stored here.
            from app.services.yolo_detector import set_tracking_box

            tracking_sink = set_tracking_box
        self._tracking_sink = tracking_sink
        self._profiles = load_profiles(config.proximity_profiles_json)

        self._lock = threading.RLock()
        self._generation = 0
        self._mission: Mission | None = None
        self._seq = 0
        self._state = self._blank_state(None)
        self._shutdown = False
        self._threads: list[threading.Thread] = []
        self._executor: ThreadPoolExecutor | None = None

        self._motion = Motion(
            lock=self._lock,
            drive=drive,
            vision=vision,
            config=config,
            is_current=self._is_current,
            current_phase=lambda: self._state.phase,
        )

    # ------------------------------------------------------------------
    # Public API -- thread-safe, never blocks on inference, sleeps or frames
    # ------------------------------------------------------------------

    def start_search(self, search_id: str, *, replace: bool = False) -> MovementState:
        """Start the scan mission for a SEARCHING search.

        Same id already active -> no-op. Another search active -> RoverBusyError,
        or with ``replace=True`` that search is cancelled exactly like
        ``cancel_search`` (reason "replaced") and this one starts.
        """
        with self._lock:
            self._ensure_running()
            active = self._live_mission()
            if active is not None and active.search_id == search_id:
                return self._copy_state()

            record = self._store.load_search(search_id)
            if record is None:
                raise SearchNotFoundError(f"search {search_id} not found")
            if record.status != SearchStatus.SEARCHING:
                raise InvalidTransitionError(f"search is {record.status.value}; only a SEARCHING search can be started")

            if active is not None:
                if not replace:
                    raise RoverBusyError(active.search_id, active.target_text)
                self._cancel_search_locked(active.search_id, reason="replaced")
            return self._begin_mission_locked(record, SCAN)

    def resume(self, search_id: str) -> MovementState:
        """Restart movement for a search whose mission is gone (e-stop, manual
        override, restart, target lost...). SEARCHING -> scan mission;
        FOUND with an accepted candidate and not yet arrived -> approach."""
        with self._lock:
            self._ensure_running()
            active = self._live_mission()
            if active is not None and active.search_id == search_id:
                return self._copy_state()

            record = self._store.load_search(search_id)
            if record is None:
                raise SearchNotFoundError(f"search {search_id} not found")
            if active is not None:
                raise RoverBusyError(active.search_id, active.target_text)

            if record.status == SearchStatus.SEARCHING:
                return self._begin_mission_locked(record, SCAN)
            if record.status == SearchStatus.FOUND:
                if self._state_for_locked(search_id).phase == MovementPhase.ARRIVED:
                    raise InvalidTransitionError("the rover already arrived at this item")
                candidate = self._store.accepted_candidate(search_id)
                if candidate is None:
                    raise InvalidTransitionError("search is FOUND but has no accepted candidate to approach")
                if candidate.box is None:
                    # Identification only: the user said yes to a full frame,
                    # not to a located object. Nothing seen later can be tied
                    # to that answer, so there is nothing to drive toward.
                    raise InvalidTransitionError(
                        "the accepted candidate has no location, so there is nothing to approach; start a new search"
                    )
                accepted = AcceptedTarget(
                    candidate_id=candidate.candidate_id,
                    description=candidate.description,
                    box=candidate.box,
                    crop_path=candidate.crop_path,
                )
                return self._begin_mission_locked(record, APPROACH, accepted)
            if record.status == SearchStatus.CANDIDATE_PENDING:
                raise InvalidTransitionError("answer the pending candidate before resuming")
            raise InvalidTransitionError(f"search is {record.status.value} and cannot be resumed")

    def accept_candidate(self, search_id: str, candidate_id: str) -> MovementState:
        """User said "yes, that's it": CANDIDATE_PENDING -> FOUND (+ Find row).

        When the live mission is waiting on exactly this candidate it goes on
        to the approach. With no live mission (after an e-stop, or a backend
        restart while the question was on screen) only the recognition
        lifecycle is updated and the rover stays where it is -- movement then
        needs an explicit ``resume``. When in doubt, the wheels stay stopped.
        """
        with self._lock:
            self._store.accept_candidate(search_id, candidate_id)  # compare-and-set; raises
            self._publish(search_id, {"type": "candidate_accepted", "payload": {"candidate_id": candidate_id, "status": "FOUND"}})
            mission = self._waiting_mission(search_id, candidate_id)
            if mission is not None:
                mission.pending_candidate_id = None
                mission.decision = ("accepted", candidate_id)
                self._apply_state_locked(MovementPhase.CHECKING, "Making sure I can still see it")
                mission.decision_event.set()
            return self._state_for_locked(search_id)

    def reject_candidate(self, search_id: str, candidate_id: str) -> MovementState:
        """User said "no": candidate REJECTED, search back to SEARCHING. A live
        mission resumes scanning; without one nothing moves (see accept)."""
        with self._lock:
            self._store.reject_candidate(search_id, candidate_id)  # compare-and-set; raises
            self._publish(
                search_id, {"type": "candidate_rejected", "payload": {"candidate_id": candidate_id, "status": "SEARCHING"}}
            )
            mission = self._waiting_mission(search_id, candidate_id)
            if mission is not None:
                mission.pending_candidate_id = None
                mission.decision = ("rejected", candidate_id)
                self._apply_state_locked(MovementPhase.SCANNING, "Okay, not that one — I'll keep looking")
                mission.decision_event.set()
            return self._state_for_locked(search_id)

    def stop(self, reason: StopReason = StopReason.USER_STOP) -> MovementState:
        """E-stop: halt NOW and invalidate the mission. The search itself is
        untouched, so it can be resumed. Always stops the wheels, even when no
        mission is running."""
        with self._lock:
            self._invalidate_locked(reason, _stop_message(reason))
            return self._copy_state()

    def stop_search(self, search_id: str, reason: StopReason = StopReason.USER_STOP) -> MovementState:
        """Halt the mission of ``search_id`` if (and only if) it is the active
        one. Does not change ``Search.status``."""
        with self._lock:
            active = self._live_mission()
            if active is not None and active.search_id == search_id:
                self._invalidate_locked(reason, _stop_message(reason))
            return self._state_for_locked(search_id)

    def cancel_search(self, search_id: str) -> bool:
        """Halt the rover FIRST if this search is the active one, then mark it
        CANCELLED. Idempotent. Returns False for an unknown search."""
        with self._lock:
            return self._cancel_search_locked(search_id, reason="cancelled")

    def manual_override(self) -> None:
        """Call before any manual drive command: a human on the controls always
        wins. No-op when no mission is running."""
        with self._lock:
            if self._live_mission() is None:
                return
            self._invalidate_locked(StopReason.MANUAL_OVERRIDE, _stop_message(StopReason.MANUAL_OVERRIDE))

    def state_for(self, search_id: str) -> MovementState:
        """Live state when this search is the active/last one, else what was
        persisted. A persisted ACTIVE phase with no live mission means the
        process restarted mid-mission: reported as stopped / interrupted."""
        with self._lock:
            if self._state.search_id == search_id:
                return self._copy_state()
        # The UI polls this for every open search page. Read the database
        # WITHOUT the lock so a slow disk can never stand between the user and
        # stop(); then re-check, in case a mission for this search began meanwhile.
        record = self._store.load_movement(search_id)
        with self._lock:
            return self._state_for_locked(search_id, record)

    def can_resume(self, search_id: str) -> bool:
        """Whether ``resume(search_id)`` would start a mission right now (for
        the API's ``can_resume`` flag). Never raises, never moves anything."""
        with self._lock:
            if self._shutdown or self._live_mission() is not None:
                return False  # already running, or the rover is busy with another search
        record = self._store.load_search(search_id)
        if record is None:
            return False
        if record.status == SearchStatus.SEARCHING:
            return True
        if record.status == SearchStatus.FOUND:
            if self.state_for(search_id).phase == MovementPhase.ARRIVED:
                return False
            candidate = self._store.accepted_candidate(search_id)
            # A box-less acceptance is identification only: never approachable.
            return candidate is not None and candidate.box is not None
        return False

    def active_search_id(self) -> str | None:
        with self._lock:
            mission = self._live_mission()
            return None if mission is None else mission.search_id

    def snapshot(self) -> dict:
        with self._lock:
            mission = self._live_mission()
            return {
                "active_search_id": None if mission is None else mission.search_id,
                "active_target_text": None if mission is None else mission.target_text,
                "generation": self._generation,
                "movement": self._copy_state().to_dict(),
                "vision_mode": self._vision_mode(),
                "driver": self._driver_name(),
                "shutdown": self._shutdown,
            }

    def shutdown(self) -> None:
        """Stop the mission and the motors, then join the thread (bounded).
        Idempotent; afterwards the controller refuses new missions."""
        with self._lock:
            self._shutdown = True
            self._invalidate_locked(StopReason.SHUTDOWN, _stop_message(StopReason.SHUTDOWN))
            threads = list(self._threads)
            executor, self._executor = self._executor, None

        # Outside the lock: never join or wait while holding it.
        deadline = time.monotonic() + _JOIN_BUDGET_SECONDS
        for thread in threads:
            if thread is threading.current_thread():
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                logger.warning("mission thread %s did not exit within the shutdown budget", thread.name)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def join_missions(self, timeout: float = _JOIN_BUDGET_SECONDS) -> bool:
        """Wait for every mission thread started so far to exit. True when none
        is left alive. (Used by tests and the demo; never call under a lock.)"""
        with self._lock:
            threads = list(self._threads)
        deadline = time.monotonic() + timeout
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not any(t.is_alive() for t in threads)

    # ------------------------------------------------------------------
    # Generation token
    # ------------------------------------------------------------------

    def _is_current(self, mission: Mission) -> bool:
        """Call with ``_lock`` held."""
        return (
            mission is self._mission
            and mission.generation == self._generation
            and not mission.cancel_event.is_set()
            and not mission.finished
        )

    def _live_mission(self) -> Mission | None:
        mission = self._mission
        return mission if mission is not None and self._is_current(mission) else None

    def _waiting_mission(self, search_id: str, candidate_id: str) -> Mission | None:
        mission = self._live_mission()
        if mission is None or mission.search_id != search_id or mission.pending_candidate_id != candidate_id:
            return None
        return mission

    def _ensure_current(self, mission: Mission) -> None:
        with self._lock:
            if not self._is_current(mission):
                raise MissionCancelled()

    def _ensure_running(self) -> None:
        if self._shutdown:
            raise RoverError("the rover controller is shut down")

    def _invalidate_locked(self, reason: StopReason, message: str) -> Mission | None:
        """The one invalidation path. Order matters: bump generation -> set the
        old mission's cancel event -> stop the wheels -> state -> persist /
        publish. Returns at once; never joins, never waits for inference."""
        live = self._live_mission()
        self._generation += 1
        stale = self._mission
        if stale is not None:
            stale.cancel_event.set()
            stale.decision_event.set()  # only to wake a mission parked on the question
        self._mission = None
        self._safe_stop()
        if live is not None:
            self._safe_tracking(None)
            self._apply_state_locked(MovementPhase.STOPPED, message, reason.value)
            logger.info("search %s: mission invalidated (%s)", live.search_id, reason.value)
        return live

    def _cancel_search_locked(self, search_id: str, *, reason: str) -> bool:
        active = self._live_mission()
        if active is not None and active.search_id == search_id:
            stop_reason = StopReason.REPLACED if reason == "replaced" else StopReason.CANCELLED
            self._invalidate_locked(stop_reason, _stop_message(stop_reason))

        was_active = active is not None and active.search_id == search_id
        changed = self._store.cancel_search(search_id)
        if changed is None:
            return False
        if changed or was_active:
            logger.info("search %s: status -> CANCELLED (%s)", search_id, reason)
            self._publish(
                search_id,
                {"type": "search_cancelled", "payload": {"status": "CANCELLED", "reason": reason}, "terminal": True},
            )
        return True

    def _begin_mission_locked(
        self, record: SearchRecord, kind: str, accepted: AcceptedTarget | None = None
    ) -> MovementState:
        self._generation += 1
        stale = self._mission
        if stale is not None:
            stale.cancel_event.set()
            stale.decision_event.set()
        self._mission = None
        self._threads = [t for t in self._threads if t.is_alive()]
        self._state = self._blank_state(record.search_id)

        if vision_is_unsafe(self._vision, self._drive):
            # Simulated detections must never reach real motors: no mission,
            # no thread, not a single drive call.
            logger.error("search %s: refusing to start -- simulated vision with %r driver", record.search_id, self._driver_name())
            self._apply_state_locked(MovementPhase.ERROR, UNSAFE_VISION_MESSAGE, StopReason.UNSAFE_VISION.value, force=True)
            return self._copy_state()

        try:
            # Every mission starts from a commanded stop (a manual command may
            # still be running), which also anchors its first fresh frame.
            self._drive.stop()
        except Exception:  # noqa: BLE001
            logger.exception("search %s: could not stop the wheels; not starting", record.search_id)
            self._apply_state_locked(MovementPhase.ERROR, _INTERNAL_ERROR_MESSAGE, StopReason.INTERNAL_ERROR.value, force=True)
            return self._copy_state()

        mission = Mission(
            search_id=record.search_id,
            generation=self._generation,
            kind=kind,
            target_text=record.target_text,
            accepted=accepted,
            phrase=spoken_target(record.target_text),
        )
        self._mission = mission
        if kind == SCAN:
            self._apply_state_locked(MovementPhase.SCANNING, f"Looking for {mission.phrase}…", force=True)
        else:
            self._apply_state_locked(MovementPhase.CHECKING, "Making sure I can still see it", force=True)

        thread = threading.Thread(
            target=self._run_mission, args=(mission,), daemon=True, name=f"rover-mission-{record.search_id[:8]}"
        )
        mission.thread = thread
        self._threads.append(thread)
        thread.start()
        logger.info("search %s: %s mission started (generation %d)", record.search_id, kind, mission.generation)
        return self._copy_state()

    # ------------------------------------------------------------------
    # State, persistence, events (all called with ``_lock`` held)
    # ------------------------------------------------------------------

    def _vision_mode(self) -> str:
        return "simulation" if getattr(self._vision, "is_simulated", False) else "live"

    def _driver_name(self) -> str:
        try:
            return str(self._drive.driver_name)
        except Exception:  # noqa: BLE001
            return "none"

    def _blank_state(self, search_id: str | None) -> MovementState:
        return MovementState(
            search_id=search_id,
            seq=self._seq,
            scan_budget=self._config.scan_max_steps,
            vision_mode=self._vision_mode(),
            driver=self._driver_name(),
        )

    def _copy_state(self) -> MovementState:
        return replace(self._state)

    def _state_for_locked(self, search_id: str, record: MovementRecord | None = _UNREAD) -> MovementState:
        if self._state.search_id == search_id:
            return self._copy_state()

        state = self._blank_state(search_id)
        state.seq = 0
        if record is _UNREAD:
            record = self._store.load_movement(search_id)
        if record is None:
            return state
        try:
            phase = MovementPhase(record.phase)
        except ValueError:
            phase = MovementPhase.IDLE
        state.updated_at = record.updated_at
        if phase in ACTIVE_PHASES:
            # No live mission owns this search, yet the last thing written was
            # an active phase: the process died mid-mission. Nothing is moving.
            state.phase = MovementPhase.STOPPED
            state.reason = StopReason.INTERRUPTED.value
            state.message = _INTERRUPTED_MESSAGE
        else:
            state.phase, state.message, state.reason = phase, record.message, record.reason
        return state

    def _apply_state_locked(
        self, phase: MovementPhase, message: str, reason: str | None = None, *, force: bool = False
    ) -> None:
        state = self._state
        changed = (state.phase, state.message, state.reason) != (phase, message, reason)
        state.phase, state.message, state.reason = phase, message, reason
        state.active = phase in ACTIVE_PHASES and self._mission is not None
        state.driver = self._driver_name()
        if not (changed or force) or state.search_id is None:
            return

        self._seq += 1
        state.seq = self._seq
        state.updated_at = utcnow()
        try:
            self._store.save_movement(state.search_id, phase.value, message, reason, state.updated_at)
        except Exception:  # noqa: BLE001 - a DB hiccup must never break a stop
            logger.exception("search %s: could not persist movement phase %s", state.search_id, phase.value)

        event: dict[str, Any] = {"type": "movement", "payload": state.to_dict()}
        if phase == MovementPhase.ARRIVED:
            event["terminal"] = True
        self._publish(state.search_id, event)

    def _publish(self, search_id: str, event: dict[str, Any]) -> None:
        try:
            self._events.publish(search_id, event)
        except Exception:  # noqa: BLE001
            logger.exception("search %s: could not publish %s", search_id, event.get("type"))

    def _safe_stop(self) -> None:
        try:
            self._drive.stop()
        except Exception:  # noqa: BLE001 - keep going: state must still say "stopped"
            logger.exception("drive.stop() failed")

    def _safe_tracking(self, box: Box | None) -> None:
        try:
            self._tracking_sink(box)
        except Exception:  # noqa: BLE001
            logger.exception("could not update the tracking box")

    # ------------------------------------------------------------------
    # Mission side effects -- each re-checks currency under the lock
    # ------------------------------------------------------------------

    def _transition(self, mission: Mission, phase: MovementPhase, message: str) -> None:
        with self._lock:
            if not self._is_current(mission):
                raise MissionCancelled()
            self._apply_state_locked(phase, message)

    def _telemetry(self, mission: Mission, **fields: Any) -> None:
        """Numbers for the UI. Not published on their own: they ride along with
        the next phase/message change."""
        with self._lock:
            if not self._is_current(mission):
                raise MissionCancelled()
            for name, value in fields.items():
                setattr(self._state, name, value)

    def _set_tracking(self, mission: Mission, box: Box | None) -> None:
        with self._lock:
            if not self._is_current(mission):
                raise MissionCancelled()
            self._safe_tracking(box)

    def _publish_current(self, mission: Mission, event: dict[str, Any]) -> None:
        with self._lock:
            if not self._is_current(mission):
                raise MissionCancelled()
            self._publish(mission.search_id, event)

    def _pulse(self, mission: Mission, command: str, speed: float, duration: float) -> float:
        """One motor pulse; returns the (clamped) duration that was commanded."""
        if not self._motion.pulse(mission, command, speed, duration):
            raise MissionEnd(MovementPhase.ERROR, StopReason.INTERNAL_ERROR, _INTERNAL_ERROR_MESSAGE)
        return min(duration, MAX_PULSE_SECONDS)

    def _finish(self, mission: Mission, end: MissionEnd) -> None:
        """Enter a resting phase: wheels stopped, tracking box cleared, state
        persisted and published. A no-op unless the mission is still current,
        so an old mission can never disturb a replacement."""
        with self._lock:
            if not self._is_current(mission):
                return
            self._safe_stop()
            mission.finished = True
            self._mission = None
            self._safe_tracking(None)
            self._apply_state_locked(end.phase, end.message, end.reason.value)
            logger.info("search %s: mission finished -> %s (%s)", mission.search_id, end.phase.value, end.reason.value)

    # ------------------------------------------------------------------
    # Mission thread
    # ------------------------------------------------------------------

    def _run_mission(self, mission: Mission) -> None:
        try:
            ctx = self._prepare_context(mission)
            if mission.kind == SCAN:
                self._run_scan(mission, ctx)
            else:
                self._run_resumed_approach(mission, ctx)
            # Both loops only ever leave through an exception.
            raise MissionEnd(MovementPhase.ERROR, StopReason.INTERNAL_ERROR, _INTERNAL_ERROR_MESSAGE)
        except MissionCancelled:
            logger.info("search %s: mission thread (generation %d) cancelled", mission.search_id, mission.generation)
        except MissionEnd as end:
            self._finish(mission, end)
        except Exception:  # noqa: BLE001
            logger.exception("search %s: mission crashed", mission.search_id)
            self._finish(mission, MissionEnd(MovementPhase.ERROR, StopReason.INTERNAL_ERROR, _INTERNAL_ERROR_MESSAGE))
        finally:
            # Generation-safe cleanup: does nothing unless this mission is
            # somehow still the current one.
            self._finish(mission, MissionEnd(MovementPhase.ERROR, StopReason.INTERNAL_ERROR, _INTERNAL_ERROR_MESSAGE))

    def _prepare_context(self, mission: Mission) -> _Context:
        record = self._store.load_search(mission.search_id)
        if record is None:
            raise MissionEnd(MovementPhase.ERROR, StopReason.INTERNAL_ERROR, _INTERNAL_ERROR_MESSAGE)

        category, landmarks, described = self._resolve_target(record.target_text)
        if described:
            # Lead-ins stripped by the comprehension layer ("find my bottle" ->
            # "my bottle"): nicer to say back. Display only.
            mission.phrase = spoken_target(described)
        references = [data for data in map(self._store.read_media, record.reference_paths) if data is not None]

        # Rejection memory and the candidate budget belong to the search, not
        # the mission, so a resumed search does not ask about the same object.
        for rejected in self._store.rejected_candidates(mission.search_id):
            crop = self._store.read_media(rejected.crop_path) if rejected.box is not None else None
            mission.rejected.append(RejectedHint(description=rejected.description, crop_jpeg=crop))
        mission.candidates_presented = self._store.candidate_count(mission.search_id)

        self._telemetry(mission, category=category, candidates_presented=mission.candidates_presented)
        logger.info(
            "search %s: target %r -> category %r, landmarks %r (%s)",
            mission.search_id,
            record.target_text,
            category,
            landmarks,
            "local screening + OMNI" if category else "OMNI sampling on every stop",
        )
        return _Context(
            category=category,
            landmarks=landmarks,
            references=references,
            profile=profile_for(category, self._profiles),
        )

    def _resolve_target(self, target_text: str) -> tuple[str | None, tuple[str, ...], str | None]:
        """(category, landmarks, cleaned description). Category is only a hint
        for local screening; OMNI always gets the full text. Landmarks are
        never used for gating."""
        try:
            resolver = self._resolver
            if resolver is None:
                from app.services.comprehension import resolve_target as resolver  # lazy: optional seam
            resolved = resolver(target_text)
        except Exception:  # noqa: BLE001 - no category just means "ask OMNI on every stop"
            logger.exception("could not resolve target %r; using OMNI sampling", target_text)
            return None, (), None
        category = getattr(resolved, "category", None)
        if not isinstance(category, str) or not category.strip():
            category = None
        landmarks = tuple(str(item) for item in (getattr(resolved, "landmarks", None) or ()))
        described = getattr(resolved, "target_text", None)
        if not isinstance(described, str) or not described.strip():
            described = None
        return category, landmarks, described

    def _check_search_status(self, mission: Mission, expected: SearchStatus) -> None:
        """The database is the source of truth for the recognition lifecycle.
        If something changed the search behind the controller's back, stop."""
        status = self._store.search_status(mission.search_id)
        if status == expected:
            return
        if status == SearchStatus.CANCELLED:
            raise MissionEnd(MovementPhase.STOPPED, StopReason.CANCELLED, _stop_message(StopReason.CANCELLED))
        raise MissionEnd(
            MovementPhase.STOPPED, StopReason.INTERRUPTED, "The search changed while I was working — stopping"
        )

    # --- frames -------------------------------------------------------------

    def _fresh_frame(self, mission: Mission) -> FramePacket:
        """A frame newer than the last one used AND captured after the wheels
        stopped and settled. Retries while stationary; too many consecutive
        failures end the mission in ERROR / STALE_FRAMES."""
        cfg = self._config
        while True:
            self._ensure_current(mission)
            not_before = mission.last_motion_stopped_at + cfg.settle_seconds
            try:
                packet = self._camera.wait_for_frame(
                    after_seq=mission.last_used_seq,
                    not_before=not_before,
                    timeout=cfg.frame_timeout_seconds + cfg.settle_seconds,
                    cancel=mission.cancel_event,
                )
            except Exception:  # noqa: BLE001
                logger.exception("search %s: camera wait failed", mission.search_id)
                packet = None
            if mission.cancel_event.is_set():
                raise MissionCancelled()

            if packet is not None and (
                packet.frame is None or packet.seq <= mission.last_used_seq or packet.captured_at < not_before
            ):
                # Defence in depth: never trust the port to have honoured the
                # freshness contract.
                logger.warning("search %s: camera returned a stale frame (seq %s)", mission.search_id, packet.seq)
                packet = None

            if packet is not None:
                mission.frame_failures = 0
                mission.last_used_seq = packet.seq
                return packet

            mission.frame_failures += 1
            logger.warning(
                "search %s: no fresh frame (%d/%d)", mission.search_id, mission.frame_failures, cfg.max_frame_failures
            )
            if mission.frame_failures >= cfg.max_frame_failures:
                raise MissionEnd(
                    MovementPhase.ERROR, StopReason.STALE_FRAMES, "My camera stopped sending pictures — stopping"
                )

    # --- vision -------------------------------------------------------------

    def _screen(self, packet: FramePacket, category: str | None) -> list[LocalDetection] | None:
        if category is None or not self._config.local_detection_enabled:
            return None
        try:
            detections = self._vision.screen(packet, category)
        except Exception:  # noqa: BLE001 - screening is only an accelerator
            logger.exception("local screening raised; falling back to OMNI sampling")
            return None
        if detections is None:
            return None
        valid = [det for det in detections if geometry.is_valid_box(det.box)]
        valid.sort(key=lambda det: det.confidence, reverse=True)
        return valid

    def _get_executor(self) -> ThreadPoolExecutor:
        with self._lock:
            self._ensure_running()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rover-infer")
            return self._executor

    def _infer(self, mission: Mission, fn: Callable[[], IdentifyResult]) -> IdentifyResult | None:
        """Run one identification with a deadline, cancellably. The wheels are
        stopped for the whole wait. Returns None after a (counted, published)
        failure; raises MissionCancelled the moment the mission is invalidated
        -- the HTTP call may still finish later, but nobody is listening."""
        cfg = self._config
        self._ensure_current(mission)
        try:
            future: Future = self._get_executor().submit(fn)
        except (RuntimeError, RoverError):  # executor or controller shut down
            raise MissionCancelled() from None

        done = threading.Event()
        future.add_done_callback(lambda _future: done.set())
        deadline = time.monotonic() + cfg.inference_timeout_seconds
        while not done.is_set():
            if mission.cancel_event.is_set():
                future.cancel()
                raise MissionCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done.wait(min(_INFER_SLICE_SECONDS, remaining))
        if mission.cancel_event.is_set():
            raise MissionCancelled()

        error: str | None = None
        result: IdentifyResult | None = None
        if not done.is_set():
            future.cancel()
            error = f"vision call timed out after {cfg.inference_timeout_seconds:g}s"
        else:
            try:
                result = future.result(timeout=0)
            except Exception as exc:  # noqa: BLE001
                error = str(exc) or exc.__class__.__name__
            else:
                if not isinstance(result, IdentifyResult) or result.likelihood not in _LIKELIHOODS:
                    error, result = "vision returned a malformed result", None

        if error is None:
            if mission.vision_errors:
                with self._lock:
                    if self._is_current(mission):
                        self._publish(mission.search_id, {"type": "vision_recovered", "payload": {}})
            mission.vision_errors = 0
            return result

        logger.warning("search %s: vision error: %s", mission.search_id, error)
        with self._lock:
            if not self._is_current(mission):
                raise MissionCancelled()
            mission.vision_errors += 1
            self._publish(mission.search_id, {"type": "vision_error", "payload": {"message": error}})
        if mission.vision_errors >= cfg.max_vision_errors:
            raise MissionEnd(
                MovementPhase.ERROR, StopReason.INFERENCE_FAILED, "I can't reach my vision service — stopping"
            )
        if mission.cancel_event.wait(min(2 ** mission.vision_errors, 30)):
            raise MissionCancelled()
        return None

    def _identify(
        self, mission: Mission, packet: FramePacket, ctx: _Context, *, confirmed_crop: bytes | None = None
    ) -> tuple[IdentifyResult, FramePacket]:
        """Identification with stationary retries: a failed call never leads to
        motion, only to another look from the same spot (bounded by
        ``max_vision_errors``)."""
        while True:
            request = IdentifyRequest(
                target_text=mission.target_text,
                reference_images=list(ctx.references),
                confirmed_crop=confirmed_crop,
                rejected=list(mission.rejected),
            )
            # Bind now: an abandoned (timed-out) call must keep its own frame.
            result = self._infer(mission, lambda frame=packet, request=request: self._vision.identify(frame, request))
            if result is not None:
                logger.info(
                    "search %s: identify on frame %d -> found=%s likelihood=%s box=%s (%.2fs)",
                    mission.search_id, packet.seq, result.found, result.likelihood, result.box, result.latency_seconds,
                )
                return result, packet
            packet = self._fresh_frame(mission)

    # --- scan -----------------------------------------------------------------

    def _run_scan(self, mission: Mission, ctx: _Context) -> None:
        cfg = self._config
        looking = f"Looking for {mission.phrase}…"
        turn_right = cfg.scan_direction.strip().lower() != "left"
        command = "turn_right" if turn_right else "turn_left"
        steps_since_omni = cfg.omni_check_every_steps  # so the very first view gets an OMNI check

        while True:
            self._check_search_status(mission, SearchStatus.SEARCHING)
            self._transition(mission, MovementPhase.SCANNING, looking)
            packet = self._fresh_frame(mission)

            detections = self._screen(packet, ctx.category)
            if detections is not None:
                self._set_tracking(mission, detections[0].box if detections else None)
                call_omni = bool(detections) or steps_since_omni >= cfg.omni_check_every_steps
            else:
                call_omni = True

            if call_omni:
                self._transition(mission, MovementPhase.CHECKING, "Checking what I can see")
                result, packet = self._identify(mission, packet, ctx)
                steps_since_omni = 0
                if result.found and search_worker._meets_threshold(result.likelihood, cfg.candidate_likelihood_threshold):
                    outcome = self._handle_match(mission, packet, result)
                    if isinstance(outcome, AcceptedTarget):
                        self._approach(mission, ctx, outcome)
                        return
                    if outcome == "rejected":
                        # Same view, one more look with the new hint: the real
                        # target may be standing right next to the rejected one.
                        continue

            if mission.scan_steps >= cfg.scan_max_steps:
                raise MissionEnd(
                    MovementPhase.EXHAUSTED, StopReason.SCAN_BUDGET, "I looked all around and couldn't find it"
                )

            self._transition(mission, MovementPhase.SCANNING, looking)
            duration = self._pulse(mission, command, cfg.turn_speed, cfg.scan_pulse_seconds)
            mission.scan_steps += 1
            steps_since_omni += 1
            # Approximate: degrees-per-second x time, no encoders behind it.
            mission.heading_est += (1 if turn_right else -1) * cfg.turn_degrees_per_second * duration
            self._telemetry(mission, scan_steps=mission.scan_steps)

    def _handle_match(self, mission: Mission, packet: FramePacket, result: IdentifyResult) -> AcceptedTarget | str:
        cfg = self._config
        box = result.box
        if box is not None and not geometry.is_valid_box(box):
            logger.warning("search %s: ignoring match with invalid box %r", mission.search_id, box)
            return "skipped"

        bearing: float | None = None
        if box is not None:
            bearing = geometry.bearing_of(box, mission.heading_est, cfg.camera_hfov_degrees)
            # Bearings are only comparable while the rover has rotated in place.
            if not mission.translated and any(
                geometry.angle_difference(bearing, rejected) <= cfg.reject_bearing_tolerance_degrees
                for rejected in mission.rejected_bearings
            ):
                logger.info("search %s: suppressing previously rejected object at ~%.0f°", mission.search_id, bearing)
                self._publish_current(
                    mission,
                    {
                        "type": "candidate_suppressed",
                        "payload": {"description": result.description, "reason": "previously_rejected"},
                    },
                )
                return "suppressed"

        if mission.candidates_presented >= cfg.max_candidates:
            raise MissionEnd(
                MovementPhase.EXHAUSTED,
                StopReason.CANDIDATE_BUDGET,
                "I've shown you everything I could find — stopping here",
            )

        candidate_id, crop_path = self._present_candidate(mission, packet, result, box)
        decision = self._await_decision(mission, candidate_id)

        if decision == "accepted":
            crop = self._store.read_media(crop_path) if box is not None else None
            return AcceptedTarget(
                candidate_id=candidate_id, description=result.description, box=box, crop_path=crop_path, crop_jpeg=crop
            )

        crop = self._store.read_media(crop_path) if box is not None else None
        mission.rejected.append(RejectedHint(description=result.description, crop_jpeg=crop))
        if bearing is not None:
            mission.rejected_bearings.append(bearing)
        self._set_tracking(mission, None)
        return "rejected"

    def _present_candidate(
        self, mission: Mission, packet: FramePacket, result: IdentifyResult, box: Box | None
    ) -> tuple[str, str]:
        """Save the images, then compare-and-set SEARCHING -> CANDIDATE_PENDING
        under the lock. If the mission is no longer current (or the search no
        longer SEARCHING) nothing is written and the images are deleted."""
        resized, scene_jpeg = result.resized_frame, result.scene_jpeg
        if resized is None or scene_jpeg is None:
            resized, scene_jpeg = search_worker._prepare_frame(packet.frame)
        # found=true with no box is allowed by the schema: the question may be
        # asked (crop = full frame, as before) but it never authorises movement.
        crop_bytes = search_worker._crop_jpeg(resized, box) if box is not None else None

        self._ensure_current(mission)
        # Files are written outside the lock so stop() never waits on disk I/O.
        full_path = search_worker._save_search_image(mission.search_id, scene_jpeg, "full")
        crop_path = full_path
        committed = False
        try:
            if crop_bytes is not None:
                crop_path = search_worker._save_search_image(mission.search_id, crop_bytes, "crop")

            with self._lock:
                if not self._is_current(mission):
                    raise MissionCancelled()
                candidate_id = self._store.create_candidate(
                    mission.search_id,
                    full_image_path=full_path,
                    crop_path=crop_path,
                    box=box,
                    description=result.description,
                )
                if candidate_id is None:
                    # Still the current mission, yet the search is no longer
                    # SEARCHING: someone changed it behind our back. Nothing
                    # was written; say why we stop (cancelled vs. anything else).
                    self._check_search_status(mission, SearchStatus.SEARCHING)
                    raise MissionEnd(
                        MovementPhase.STOPPED, StopReason.INTERRUPTED, "The search is no longer active — stopping"
                    )
                committed = True
                mission.pending_candidate_id = candidate_id
                mission.decision = None
                mission.decision_event.clear()
                mission.candidates_presented += 1
                mission.moved_since_candidate = False
                self._state.candidates_presented = mission.candidates_presented
                self._safe_tracking(box)
                self._publish(
                    mission.search_id,
                    {
                        "type": "candidate_found",
                        "payload": {
                            "candidate_id": candidate_id,
                            "image_url": f"/media/{full_path}",
                            "crop_url": f"/media/{crop_path}",
                            "box": box,
                            "description": result.description,
                        },
                    },
                )
                self._apply_state_locked(MovementPhase.WAITING_FOR_CONFIRMATION, f"Is this {mission.phrase}?")
            logger.info("search %s: candidate %s presented, status -> CANDIDATE_PENDING", mission.search_id, candidate_id)
            return candidate_id, crop_path
        finally:
            if not committed:
                self._store.delete_media(full_path)
                if crop_path != full_path:
                    self._store.delete_media(crop_path)

    def _await_decision(self, mission: Mission, candidate_id: str) -> str:
        """Park until the user answers. Wheels stopped, no timeout; wakes at
        once on accept / reject / any invalidation."""
        while True:
            mission.decision_event.wait(_DECISION_SLICE_SECONDS)
            with self._lock:
                if not self._is_current(mission):
                    raise MissionCancelled()
                if mission.decision is not None and mission.decision[1] == candidate_id:
                    verdict = mission.decision[0]
                    mission.decision = None
                    mission.decision_event.clear()
                    return verdict
                mission.decision_event.clear()

    # --- approach -------------------------------------------------------------

    def _run_resumed_approach(self, mission: Mission, ctx: _Context) -> None:
        accepted = mission.accepted
        if accepted is None:
            raise MissionEnd(MovementPhase.ERROR, StopReason.INTERNAL_ERROR, _INTERNAL_ERROR_MESSAGE)
        # Anything may have happened since the candidate frame (manual driving,
        # a restart): its box says nothing about where the target is now.
        mission.moved_since_candidate = True
        self._approach(mission, ctx, accepted)

    def _approach(self, mission: Mission, ctx: _Context, accepted: AcceptedTarget) -> None:
        cfg = self._config
        target = mission.phrase
        self._check_search_status(mission, SearchStatus.FOUND)
        if accepted.box is None:
            # A box-less acceptance is identification only. The user said yes
            # to a full frame, not to a located object, so no box OMNI produces
            # later can be tied to that answer: end here, before any look or
            # pulse. (resume() refuses the same case up front.)
            raise self._lost(target, saw_without_box=1, saw_invalid=0)
        if accepted.crop_jpeg is None:
            accepted.crop_jpeg = self._store.read_media(accepted.crop_path)
        self._transition(mission, MovementPhase.CHECKING, "Making sure I can still see it")

        track = self._reacquire(mission, ctx, accepted)
        self._report_track(mission, track)

        steering = ApproachSteering(cfg)
        started = time.monotonic()
        forward_stall = turn_stall = 0
        progress_width, progress_height = geometry.width(track.box), geometry.height(track.box)
        progress_offset = abs(geometry.offset_from_center(track.box))
        while True:
            self._check_search_status(mission, SearchStatus.FOUND)

            if steering.arrival(track.box, ctx.profile):
                self._transition(mission, MovementPhase.CHECKING, "I think I'm close — double-checking")
                if self._confirm_arrival(mission, ctx, track, steering):
                    raise MissionEnd(MovementPhase.ARRIVED, StopReason.PROXIMITY_CONFIRMED, f"I'm next to {target}")
                continue  # the evidence did not hold; track now holds the newer observation

            if (
                time.monotonic() - started > cfg.approach_max_seconds
                or mission.approach_pulses >= cfg.approach_max_pulses
            ):
                raise MissionEnd(
                    MovementPhase.STOPPED, StopReason.APPROACH_BUDGET, "I couldn't get there in time — stopping"
                )

            before = track.box
            offset = 2.0 * geometry.offset_from_center(before)  # -1..1, not -0.5..0.5
            slow_profile = replace(ctx.profile,
                arrive_width=ctx.profile.arrive_width * cfg.arrival_enter_scale,
                arrive_height=ctx.profile.arrive_height * cfg.arrival_enter_scale)
            slow = in_slow_zone(before, slow_profile, cfg.slow_zone_fraction)
            wheels = steering.wheels(offset, slow=slow)
            kind = "turn" if wheels.pivot else "forward"
            duration = steering.pulse_seconds(wheels, slow=slow)
            # The current command is held only for this lease, then stopped
            # before recognition. A missing or slow observation cannot extend it.
            duration = min(duration, cfg.lost_target_hold_seconds, MAX_PULSE_SECONDS)
            message = (f"Turning to face {target}" if wheels.pivot else
                       "Almost there — slowing down" if slow else f"Moving closer to {target}")
            self._transition(mission, MovementPhase.CENTERING if wheels.pivot else MovementPhase.APPROACHING, message)
            if not self._motion.pulse(mission, wheels.label, max(abs(wheels.left), abs(wheels.right)),
                                      duration, wheels=(wheels.left, wheels.right)):
                raise MissionEnd(MovementPhase.ERROR, StopReason.INTERNAL_ERROR, _INTERNAL_ERROR_MESSAGE)
            # Same approximate calibration as before, scaled to the actual
            # differential command. Association still checks the next image.
            delta = ((wheels.left - wheels.right) / 2 / max(cfg.turn_speed, 1e-6)
                     * cfg.turn_degrees_per_second * duration)
            mission.heading_est += delta
            predicted = geometry.predict_box_after_turn(before, delta, cfg.camera_hfov_degrees)

            mission.approach_pulses += 1
            track.pulses_since_omni += 1
            self._observe(mission, ctx, track, predicted, force_omni=False)
            self._report_track(mission, track)

            # Progress is judged only from what the camera shows afterwards --
            # a timed pulse measures neither distance nor angle.
            fraction = cfg.min_progress_fraction
            if kind == "forward":
                grew = geometry.width(track.box) >= progress_width * (1 + fraction) or geometry.height(
                    track.box
                ) >= progress_height * (1 + fraction)
                forward_stall = 0 if grew else forward_stall + 1
                if grew:
                    progress_width = max(progress_width, geometry.width(track.box))
                    progress_height = max(progress_height, geometry.height(track.box))
                turn_stall = 0  # driving forward legitimately changes the offset
                progress_offset = abs(geometry.offset_from_center(track.box))
            else:
                shrank = abs(geometry.offset_from_center(track.box)) <= progress_offset * (1 - fraction)
                turn_stall = 0 if shrank else turn_stall + 1
                if shrank:
                    progress_offset = min(progress_offset, abs(geometry.offset_from_center(track.box)))
            if forward_stall >= cfg.stall_pulses or turn_stall >= cfg.stall_pulses:
                raise MissionEnd(MovementPhase.STOPPED, StopReason.NO_PROGRESS, "I'm not getting any closer — stopping")

    def _report_track(self, mission: Mission, track: _Track) -> None:
        with self._lock:
            if not self._is_current(mission):
                raise MissionCancelled()
            self._state.target_offset = round(geometry.offset_from_center(track.box), 4)
            self._state.target_width = round(geometry.width(track.box), 4)
            self._state.approach_pulses = mission.approach_pulses
            self._safe_tracking(list(track.box))

    def _reacquire(self, mission: Mission, ctx: _Context, accepted: AcceptedTarget) -> _Track:
        """Before the first movement: see the accepted target again, from a
        standstill, WITH a location. No box, no movement -- ever. (The caller
        has already ended the mission if the accepted candidate had no box.)"""
        cfg = self._config
        target = mission.phrase
        confirmed_crop = accepted.crop_jpeg
        attempts = saw_without_box = saw_invalid = 0

        while attempts < cfg.reacquire_attempts:
            packet = self._fresh_frame(mission)
            result, packet = self._identify(mission, packet, ctx, confirmed_crop=confirmed_crop)
            attempts += 1

            if not result.found or not search_worker._meets_threshold(result.likelihood, cfg.approach_min_likelihood):
                continue
            if result.box is None:
                saw_without_box += 1
                continue
            if not geometry.is_valid_box(result.box):
                saw_invalid += 1
                continue
            if not mission.moved_since_candidate:
                # Nothing moved since the question was asked, so the target
                # must still be where the candidate was.
                observed = LocalDetection(box=list(result.box), confidence=1.0)
                if geometry.associate([observed], accepted.box, cfg).match is None:
                    raise MissionEnd(
                        MovementPhase.TARGET_LOST,
                        StopReason.IDENTITY_UNCERTAIN,
                        f"I'm not sure which one is {target}, so I've stopped",
                    )
            return _Track(box=list(result.box), source="omni", confirmed_crop=confirmed_crop)

        raise self._lost(target, saw_without_box, saw_invalid)

    @staticmethod
    def _lost(target: str, saw_without_box: int, saw_invalid: int) -> MissionEnd:
        if saw_invalid:
            return MissionEnd(
                MovementPhase.ERROR, StopReason.INVALID_DETECTION, "I got a confusing reading from my camera — stopping"
            )
        if saw_without_box:
            return MissionEnd(
                MovementPhase.TARGET_LOST,
                StopReason.NO_BOX,
                f"I think I can see {target}, but I can't tell where it is, so I'm staying put",
            )
        return MissionEnd(
            MovementPhase.TARGET_LOST, StopReason.NOT_VISIBLE, f"I lost sight of {target} — staying where I am"
        )

    def _observe(self, mission: Mission, ctx: _Context, track: _Track, predicted: Box, *, force_omni: bool) -> None:
        """Look again from a standstill and update ``track`` in place.

        YOLO may continue the track only when exactly one detection plausibly
        continues it. Ambiguity, a miss, no screening at all, or the periodic
        revalidation all go to OMNI. A YOLO miss alone is never "lost"; an OMNI
        box that does not fit the prediction is an identity problem and stops
        the rover immediately.
        """
        cfg = self._config
        target = mission.phrase
        attempts = saw_without_box = saw_invalid = 0
        missed_since: float | None = None

        while True:
            packet = self._fresh_frame(mission)
            need_omni = force_omni or track.pulses_since_omni >= cfg.omni_revalidate_every_pulses

            if not need_omni:
                detections = self._screen(packet, ctx.category)
                if detections is None:
                    need_omni = True
                else:
                    association = geometry.associate(detections, predicted, cfg)
                    if association.match is not None and not association.ambiguous:
                        track.box, track.source = list(association.match.box), "yolo"
                        return
                    need_omni = True

            result, packet = self._identify(mission, packet, ctx, confirmed_crop=track.confirmed_crop)
            attempts += 1
            usable = result.found and search_worker._meets_threshold(result.likelihood, cfg.approach_min_likelihood)
            if usable and result.box is not None and geometry.is_valid_box(result.box):
                observed = LocalDetection(box=list(result.box), confidence=1.0)
                # A stalled turn leaves the confirmed target in its old place.
                # Accept that observation only after OMNI revalidation, then
                # let the progress budget stop the wheels.
                if (geometry.associate([observed], predicted, cfg).match is None
                        and geometry.associate([observed], track.box, cfg).match is None):
                    raise MissionEnd(
                        MovementPhase.TARGET_LOST,
                        StopReason.IDENTITY_UNCERTAIN,
                        f"I'm not sure which one is {target}, so I've stopped",
                    )
                track.box, track.source, track.pulses_since_omni = list(result.box), "omni", 0
                return
            if usable and result.box is None:
                saw_without_box += 1
            elif usable:
                saw_invalid += 1

            if missed_since is None:
                missed_since = time.monotonic()
            if (attempts >= cfg.reacquire_attempts
                    or time.monotonic() - missed_since >= cfg.lost_target_timeout_seconds):
                raise self._lost(target, saw_without_box, saw_invalid)
            force_omni = True  # keep asking OMNI from the same spot

    def _confirm_arrival(self, mission: Mission, ctx: _Context, track: _Track,
                         steering: ApproachSteering) -> bool:
        """Arrival needs ``arrival_confirmations`` consecutive stationary
        observations that all show arrival evidence, the last one validated by
        OMNI. One big box proves nothing: OMNI merely *finding* the object
        never counts as arriving."""
        needed = max(1, self._config.arrival_confirmations)
        confirmations = 1  # the observation that brought us here
        while confirmations < needed or track.source != "omni":
            final = confirmations >= needed - 1
            self._observe(mission, ctx, track, track.box, force_omni=final)
            self._report_track(mission, track)
            if not steering.arrival(track.box, ctx.profile):
                return False
            confirmations += 1
        return True


_DETERMINER = re.compile(r"(?i)^(your|the|a|an|this|that|these|those|some)\b")
# "find my bottle", "where is my phone?", "can you look for ..." -> the thing itself.
_LEAD_IN = re.compile(
    r"(?i)^(?:please|hey patch|can you|could you|would you|will you|help me(?: to)?|"
    r"i'?m looking for|i am looking for|i'?m trying to find|i need(?: to find)?|i want(?: to find)?|"
    r"i lost|i can'?t find|go (?:and )?find|find|look for|locate|search for|"
    r"where is|where are|where'?s|where did i (?:put|leave)|"
    r"bring me|get me|fetch me|show me|get(?= (?:my|the|a|an)\b))\b[\s,:]*"
)
_TRAILING = re.compile(r"(?i)[\s,]*(?:please)?[\s?.!,]*$")


def spoken_target(target_text: str) -> str:
    """The target as the rover says it back to its owner: "Find my bottle
    beside my backpack" -> "your bottle beside your backpack". Display and
    speech only -- OMNI always receives the search's full, untouched text."""
    text = " ".join((target_text or "").split())
    for _ in range(6):  # "please can you find ..." peels off one lead-in at a time
        stripped = _LEAD_IN.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    text = _TRAILING.sub("", text) or "item"
    text = re.sub(r"(?i)\bmy\b", "your", text)
    text = re.sub(r"(?i)\bmine\b", "yours", text)
    return text if _DETERMINER.match(text) else f"your {text}"


def _stop_message(reason: StopReason) -> str:
    return {
        StopReason.USER_STOP: "Stopped",
        StopReason.CANCELLED: "Search cancelled",
        StopReason.MANUAL_OVERRIDE: "Manual control took over — search paused",
        StopReason.REPLACED: "Stopped for a new search",
        StopReason.SHUTDOWN: "Shutting down",
    }.get(reason, "Stopped")


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------

_controller: RoverController | None = None
_controller_lock = threading.Lock()


def _build_default_controller() -> RoverController:
    if settings.rover_simulation:
        # First thing, before any service is touched: simulated detections
        # must never be wired to real motors (RuntimeError unless MOTOR_DRIVER=sim).
        from app.rover.simulation import build_simulation_ports, ensure_simulation_allowed

        ensure_simulation_allowed()

    from app.services.camera_service import camera_service
    from app.services.drive_service import drive_service

    config = RoverConfig.from_settings()
    if settings.rover_simulation:
        drive, camera, vision = build_simulation_ports(drive_service, camera_service, config)
        logger.warning("ROVER_SIMULATION is on: detections are SIMULATED, motors are the sim driver")
    else:
        from app.rover.vision import LiveVision

        drive, camera, vision = drive_service, camera_service, LiveVision(config)
    return RoverController(drive, camera, vision, config)


def get_rover_controller() -> RoverController:
    """The controller wired to the real drive/camera services (lazy)."""
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = _build_default_controller()
        return _controller


def set_rover_controller(controller: RoverController | None) -> None:
    """Install (or with None: forget) the process-wide controller. For tests."""
    global _controller
    with _controller_lock:
        _controller = controller
