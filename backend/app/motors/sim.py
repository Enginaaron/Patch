import logging

from app.motors.base import MotorDriver

logger = logging.getLogger(__name__)


class SimMotorDriver(MotorDriver):
    """A driver with no hardware. It records the last commanded speeds and logs
    changes, so the whole drive/search stack can be developed and demoed on a
    laptop. Swap for the GPIO driver by setting ``MOTOR_DRIVER=gpio``.
    """

    name = "sim"

    def __init__(self) -> None:
        self.left = 0.0
        self.right = 0.0

    def set_speeds(self, left: float, right: float) -> None:
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        if (left, right) != (self.left, self.right):
            logger.info("[sim motors] L=%+.2f R=%+.2f", left, right)
        self.left = left
        self.right = right
