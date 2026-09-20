"""Bounded motor pulses -- the only way the autonomy layer moves the wheels.

Every movement is one short pulse:

1. under the controller lock: re-check the mission is still current, re-check
   the vision/driver pairing is safe, re-check the phase allows motion, then
   start the wheels WITH a watchdog ttl (``duration + margin``) so an
   independent deadline stops them even if this thread never runs again;
2. wait out the pulse on the mission's cancel event (outside the lock, so a
   stop / cancel / override never waits for us);
3. under the lock: stop the wheels -- but only if the mission is still
   current. If it is not, whoever invalidated it already stopped the wheels,
   and a late stop from here could cut a manual drive command.

A pulse never measures distance or angle. What it achieved is judged only
from the next fresh camera frame.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from app.rover.config import RoverConfig
from app.rover.mission import Mission, MissionEnd
from app.rover.types import (
    MOVING_PHASES,
    DrivePort,
    MissionCancelled,
    MovementPhase,
    StopReason,
    VisionPort,
)

logger = logging.getLogger(__name__)

# The autonomy layer only ever rotates in place or creeps forward.
ALLOWED_COMMANDS = frozenset({"forward", "turn_left", "turn_right"})

# Hard ceiling on a single pulse regardless of configuration: a typo in .env
# must not turn "short pulse, then look" into a long blind drive.
MAX_PULSE_SECONDS = 1.5

UNSAFE_VISION_MESSAGE = "Simulated vision can't drive real motors — refusing to move"


def vision_is_unsafe(vision: VisionPort, drive: DrivePort) -> bool:
    """Simulated detections must never move real motors."""
    return bool(getattr(vision, "is_simulated", False)) and drive.driver_name != "sim"


class Motion:
    def __init__(
        self,
        *,
        lock: threading.RLock,
        drive: DrivePort,
        vision: VisionPort,
        config: RoverConfig,
        is_current: Callable[[Mission], bool],
        current_phase: Callable[[], MovementPhase],
    ) -> None:
        self._lock = lock
        self._drive = drive
        self._vision = vision
        self._config = config
        self._is_current = is_current        # must be called with the lock held
        self._current_phase = current_phase  # must be called with the lock held

    def pulse(self, mission: Mission, command: str, speed: float, duration: float) -> bool:
        """Run one pulse. Returns True when it ran to completion, False when it
        was refused without touching the motors (phase does not allow motion).
        Raises MissionCancelled when the mission was invalidated before or
        during the pulse, MissionEnd for an unsafe vision/driver pairing."""
        if command not in ALLOWED_COMMANDS:
            raise ValueError(f"autonomy may not issue drive command {command!r}")
        if not (duration > 0) or not (speed > 0):
            raise ValueError("pulse speed and duration must be positive")
        duration = min(float(duration), MAX_PULSE_SECONDS)
        speed = min(float(speed), 1.0)
        ttl = duration + self._config.pulse_watchdog_margin_seconds

        with self._lock:
            if not self._is_current(mission):
                raise MissionCancelled()
            if vision_is_unsafe(self._vision, self._drive):
                raise MissionEnd(MovementPhase.ERROR, StopReason.UNSAFE_VISION, UNSAFE_VISION_MESSAGE)
            phase = self._current_phase()
            if phase not in MOVING_PHASES:
                logger.error("search %s: refused %s pulse in phase %s", mission.search_id, command, phase.value)
                return False
            mission.moved_since_candidate = True
            if command == "forward":
                mission.translated = True
            self._drive.drive(command, speed, ttl=ttl)

        # Outside the lock: a stop/cancel sets the event and we wake at once.
        mission.cancel_event.wait(duration)

        with self._lock:
            if not self._is_current(mission):
                # The invalidator already stopped the wheels (under this same
                # lock). Anything moving now belongs to someone else.
                raise MissionCancelled()
            self._drive.stop()
            mission.last_motion_stopped_at = time.monotonic()
        return True
