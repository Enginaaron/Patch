import logging

from app.config import settings
from app.motors.base import MotorDriver

logger = logging.getLogger(__name__)


class GpioMotorDriver(MotorDriver):
    """Differential drive over a TB6612FNG dual H-bridge.

    Each motor channel has two direction pins (INx1/INx2) and one PWM pin. The
    chip is enabled by driving STBY high. Direction truth table per channel:

        IN1=1 IN2=0 -> forward
        IN1=0 IN2=1 -> reverse
        IN1=0 IN2=0 -> coast (stop)

    Speed is the PWM duty on the PWM pin. Uses ``gpiozero`` (which selects the
    ``lgpio`` backend on a Pi 5) and imports it lazily so laptop dev is fine.
    """

    name = "gpio"

    def __init__(self) -> None:
        try:
            from gpiozero import DigitalOutputDevice, PWMOutputDevice
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "gpiozero is unavailable — run `pip install gpiozero lgpio` on the "
                "Pi, or use MOTOR_DRIVER=sim"
            ) from exc

        freq = settings.motor_pwm_hz
        self._standby = DigitalOutputDevice(settings.motor_standby_pin)

        self._left_pwm = PWMOutputDevice(settings.motor_left_pwm_pin, frequency=freq)
        self._left_in1 = DigitalOutputDevice(settings.motor_left_in1_pin)
        self._left_in2 = DigitalOutputDevice(settings.motor_left_in2_pin)

        self._right_pwm = PWMOutputDevice(settings.motor_right_pwm_pin, frequency=freq)
        self._right_in1 = DigitalOutputDevice(settings.motor_right_in1_pin)
        self._right_in2 = DigitalOutputDevice(settings.motor_right_in2_pin)

        self._left_invert = settings.motor_left_invert
        self._right_invert = settings.motor_right_invert

        self._standby.on()  # enable the chip
        self.stop()
        logger.info("TB6612 GPIO motor driver ready")

    def _set_channel(self, in1, in2, pwm, speed: float, invert: bool) -> None:
        speed = max(-1.0, min(1.0, speed))
        if invert:
            speed = -speed
        if speed > 0:
            in1.on()
            in2.off()
        elif speed < 0:
            in1.off()
            in2.on()
        else:
            in1.off()
            in2.off()
        pwm.value = abs(speed)

    def set_speeds(self, left: float, right: float) -> None:
        self._set_channel(self._left_in1, self._left_in2, self._left_pwm, left, self._left_invert)
        self._set_channel(self._right_in1, self._right_in2, self._right_pwm, right, self._right_invert)

    def close(self) -> None:
        self.stop()
        try:
            self._standby.off()  # disable the chip
        except Exception:  # noqa: BLE001
            pass
        for dev in (
            self._left_pwm, self._left_in1, self._left_in2,
            self._right_pwm, self._right_in1, self._right_in2,
            self._standby,
        ):
            try:
                dev.close()
            except Exception:  # noqa: BLE001
                pass
