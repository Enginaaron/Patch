"""Autonomous visual search.

A background loop that, for the active search, repeatedly:

    look at the current frame  ->  ask OMNI where the target is  ->  steer

Steering is closed-loop on the target's horizontal position and apparent size:
rotate to scan when nothing is seen, turn to centre the target, drive forward to
close in, and stop (declare a find) once it fills enough of the frame.

Only one search runs at a time — Patch has one body. Manual teleop should call
:meth:`stop` first so the two don't fight over the wheels.
"""

import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import cv2

from app.config import settings
from app.db import get_session
from app.models import Search, SearchStatus
from app.services.camera_service import camera_service
from app.services.drive_service import drive_service
from app.services.omni_service import omni_service

logger = logging.getLogger(__name__)


@dataclass
class SearchState:
    active: bool = False
    search_id: str | None = None
    target_text: str = ""
    phase: str = "idle"  # idle | scanning | centering | approaching | found | stopped
    message: str = ""
    x: float = 0.5
    size: float = 0.0
    confidence: float = 0.0
    mode: str = "demo"
    ticks: int = 0


class SearchService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state = SearchState()

    def start(self, search_id: str, target_text: str) -> SearchState:
        self.stop()  # ensure only one loop runs
        self._stop_event.clear()
        with self._lock:
            self._state = SearchState(
                active=True,
                search_id=search_id,
                target_text=target_text,
                phase="scanning",
                message="Starting search…",
            )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("search started for %r (id=%s)", target_text, search_id)
        return self.status()

    def stop(self) -> SearchState:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
        self._thread = None
        drive_service.stop()
        with self._lock:
            if self._state.active:
                self._state.active = False
                self._state.phase = "stopped"
                self._state.message = "Search stopped."
        return self.status()

    def status(self) -> dict:
        with self._lock:
            return asdict(self._state)

    def _run(self) -> None:
        tick = 0
        while not self._stop_event.is_set():
            target = self._state.target_text
            frame = camera_service.get_frame()
            frame_jpeg = None
            if frame is not None:
                ok, buffer = cv2.imencode(".jpg", frame)
                if ok:
                    frame_jpeg = buffer.tobytes()

            loc = omni_service.locate_target(
                target_text=target, frame_jpeg=frame_jpeg, tick=tick
            )
            self._act_on(loc)

            if not self._state.active:  # found -> loop is done
                break
            tick += 1
            self._stop_event.wait(settings.search_interval_seconds)

    def _act_on(self, loc) -> None:
        weak = (not loc.found) or loc.confidence < settings.search_min_confidence

        if weak:
            phase, message = "scanning", f"Looking for {self._state.target_text}…"
            drive_service.drive("turn_right", settings.search_scan_speed)
        elif loc.done or loc.size >= settings.search_arrival_size:
            phase, message = "found", loc.note or f"Found {self._state.target_text}."
            drive_service.stop()
        else:
            offset = loc.x - 0.5
            if abs(offset) > settings.search_center_tolerance:
                phase, message = "centering", "Centering on the target…"
                command = "turn_left" if offset < 0 else "turn_right"
                drive_service.drive(command, settings.search_turn_speed)
            else:
                phase, message = "approaching", "Moving closer…"
                drive_service.drive("forward", settings.search_forward_speed)

        with self._lock:
            s = self._state
            s.phase, s.message = phase, message
            s.x, s.size, s.confidence, s.mode = loc.x, loc.size, loc.confidence, loc.mode
            s.ticks += 1
            if phase == "found":
                s.active = False

        if phase == "found":
            self._mark_found()

    def _mark_found(self) -> None:
        search_id = self._state.search_id
        if not search_id:
            return
        try:
            with get_session() as session:
                search = session.get(Search, search_id)
                if search is not None:
                    search.status = SearchStatus.FOUND
                    search.ended_at = datetime.utcnow()
                    session.add(search)
                    session.commit()
        except Exception:  # noqa: BLE001
            logger.exception("failed to persist FOUND status for %s", search_id)


search_service = SearchService()
