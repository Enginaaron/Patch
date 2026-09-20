"""High-level differential drive.

Turns named commands (forward, turn_left, rotate_right, stop, …) into signed
left/right wheel speeds and pushes them to the active :class:`MotorDriver`.
Also the single source of truth for "what is Patch doing right now", which the
search loop and the UI both read.
"""

import logging
import threading

from app.config import settings
from app.motors.base import MotorDriver
from app.motors.sim import SimMotorDriver

logger = logging.getLogger(__name__)

# command -> (left, right) as fractions of the requested speed
COMMANDS: dict[str, tuple[float, float]] = {
    "forward": (1.0, 1.0),
    "backward": (-1.0, -1.0),
    "turn_left": (-1.0, 1.0),   # rotate in place, counter-clockwise
    "turn_right": (1.0, -1.0),  # rotate in place, clockwise
    "veer_left": (0.3, 1.0),    # gentle arc while moving forward
    "veer_right": (1.0, 0.3),
    "stop": (0.0, 0.0),
}


def _make_driver() -> MotorDriver:
    if settings.motor_driver == "gpio":
        try:
            from app.motors.gpio import GpioMotorDriver

            return GpioMotorDriver()
        except Exception:  # noqa: BLE001
            logger.exception("GPIO driver failed to init; falling back to sim")
    return SimMotorDriver()


class DriveService:
    def __init__(self) -> None:
        self._driver: MotorDriver | None = None
        self._lock = threading.Lock()
        self.command = "stop"
        self.left = 0.0
        self.right = 0.0

    def start(self) -> None:
        self._driver = _make_driver()
        logger.info("drive service using '%s' motor driver", self._driver.name)

    def set_speeds(self, left: float, right: float, command: str = "custom") -> None:
        cap = max(0.0, min(1.0, settings.drive_max_speed))
        left = max(-cap, min(cap, left))
        right = max(-cap, min(cap, right))
        with self._lock:
            if self._driver is not None:
                self._driver.set_speeds(left, right)
            self.left, self.right, self.command = left, right, command

    def drive(self, command: str, speed: float | None = None) -> None:
        if command not in COMMANDS:
            raise ValueError(f"unknown drive command: {command!r}")
        speed = settings.drive_max_speed if speed is None else max(0.0, min(1.0, speed))
        lf, rf = COMMANDS[command]
        self.set_speeds(lf * speed, rf * speed, command=command)

    def stop(self) -> None:
        self.drive("stop")

    def state(self) -> dict:
        driver = self._driver.name if self._driver else "none"
        return {"command": self.command, "left": self.left, "right": self.right, "driver": driver}

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


drive_service = DriveService()
