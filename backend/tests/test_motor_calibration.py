"""Verify wheel trim preserves stop, direction, and maximum power."""
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.config import Settings, settings
from app.motors.sim import SimMotorDriver
from app.services.drive_service import DriveService


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = patch.multiple(settings, motor_left_scale=0.8,
                                     motor_right_scale=1.0, drive_max_speed=0.6)
        self.config.start()
        self.addCleanup(self.config.stop)
        self.drive = DriveService()
        self.drive._driver = SimMotorDriver()

    def test_forward_reverse_and_state(self):
        for command, sign in (("forward", 1), ("backward", -1)):
            self.drive.drive(command, 0.5)
            self.assertAlmostEqual(self.drive._driver.left, sign * 0.4)
            self.assertAlmostEqual(self.drive._driver.right, sign * 0.5)
            self.assertEqual(self.drive.state()["left"], self.drive._driver.left)

    def test_stationary_wheel_and_stop_stay_zero(self):
        self.drive.set_speeds(0, 0.5)
        self.assertEqual(self.drive._driver.left, 0)
        self.drive.stop()
        self.assertEqual((self.drive._driver.left, self.drive._driver.right), (0, 0))

    def test_turns_preserve_directions(self):
        self.drive.drive("turn_left", 0.5)
        self.assertAlmostEqual(self.drive._driver.left, -0.4)
        self.assertAlmostEqual(self.drive._driver.right, 0.5)

    def test_boost_cannot_exceed_cap(self):
        with patch.object(settings, "motor_left_scale", 2):
            self.drive.set_speeds(0.5, -1)
        self.assertEqual((self.drive._driver.left, self.drive._driver.right), (0.6, -0.6))

    def test_invalid_scales_rejected(self):
        for side in ("left", "right"):
            for value in (0, -1, 2.1, float("nan"), float("inf")):
                with self.subTest(side=side, value=value):
                    with self.assertRaises(ValidationError):
                        Settings(_env_file=None, **{f"motor_{side}_scale": value})


if __name__ == "__main__":
    unittest.main()
