import logging

from app.config import settings
from app.motors.base import MotorDriver

logger = logging.getLogger(__name__)


class GpioMotorDriver(MotorDriver):
    """Differential drive over a two-motor H-bridge (TB6612 / L298N style).

    Each wheel has one PWM pin (speed) and one direction pin. This uses
    ``gpiozero`` (which picks a working pin backend on the Pi) and is imported
    lazily so nothing here breaks development on a laptop.

    NOTE: pin numbers and direction/enable wiring depend on the final board.
    Confirm ``MOTOR_*`` pins in ``.env`` and, if the board uses two direction
    pins per motor (plain L298N) rather than one, adjust ``_set_wheel``.
    """

    name = "gpio"

    def __init__(self) -> None:
        try:
            from gpiozero import DigitalOutputDevice, PWMOutputDevice
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "gpiozero is unavailable — install it on the Pi or use MOTOR_DRIVER=sim"
            ) from exc

        freq = settings.motor_pwm_hz
        self._left_pwm = PWMOutputDevice(settings.motor_left_pwm_pin, frequency=freq)
        self._left_dir = DigitalOutputDevice(settings.motor_left_dir_pin)
        self._right_pwm = PWMOutputDevice(settings.motor_right_pwm_pin, frequency=freq)
        self._right_dir = DigitalOutputDevice(settings.motor_right_dir_pin)
        logger.info("GPIO motor driver ready")

    def _set_wheel(self, pwm, direction, speed: float) -> None:
        speed = max(-1.0, min(1.0, speed))
        direction.value = 1 if speed >= 0 else 0
        pwm.value = abs(speed)

    def set_speeds(self, left: float, right: float) -> None:
        self._set_wheel(self._left_pwm, self._left_dir, left)
        self._set_wheel(self._right_pwm, self._right_dir, right)

    def close(self) -> None:
        self.stop()
        for dev in (self._left_pwm, self._left_dir, self._right_pwm, self._right_dir):
            try:
                dev.close()
            except Exception:  # noqa: BLE001
                pass
