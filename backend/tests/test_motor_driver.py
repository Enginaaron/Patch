"""Electrical output checks using gpiozero's simulated GPIO pins."""

import unittest
from unittest.mock import patch

from gpiozero import Device
from gpiozero.pins.mock import MockFactory, MockPWMPin

from app.config import Settings, settings
from app.motors.gpio import GpioMotorDriver
from app.services.drive_service import DriveService


class MotorDriverTests(unittest.TestCase):
    def setUp(self):
        Device.pin_factory = MockFactory(pin_class=MockPWMPin)
        self.config = patch("app.motors.gpio.settings", Settings(_env_file=None))
        self.config.start()
        self.driver = GpioMotorDriver()

    def tearDown(self):
        self.driver.close()
        self.config.stop()
        Device.pin_factory.close()

    def value(self, pin):
        return Device.pin_factory.pin(pin).state

    def test_wiring_forward_reverse_and_stop(self):
        self.assertTrue(self.value(16))
        self.driver.set_speeds(0.4, -0.3)
        for pin, expected in {12: 0.4, 5: 1, 6: 0, 13: 0.3, 20: 0, 21: 1}.items():
            self.assertAlmostEqual(self.value(pin), expected)
        self.driver.set_speeds(-0.4, 0.3)
        for pin, expected in {5: 0, 6: 1, 20: 1, 21: 0}.items():
            self.assertEqual(self.value(pin), expected)
        self.driver.stop()
        for pin in (12, 5, 6, 13, 20, 21):
            self.assertEqual(self.value(pin), 0)

    def test_direction_inversion(self):
        self.driver._right_invert = True
        self.driver.set_speeds(0.4, 0.4)
        self.assertEqual(self.value(5), 1)
        self.assertEqual(self.value(20), 0)
        self.assertEqual(self.value(21), 1)

    def test_turns_reach_correct_wheels(self):
        service = DriveService()
        service._driver = self.driver
        service.drive("turn_left", 0.4)
        self.assertEqual(self.value(6), 1)
        self.assertEqual(self.value(20), 1)
        service.drive("turn_right", 0.4)
        self.assertEqual(self.value(5), 1)
        self.assertEqual(self.value(21), 1)

    def test_hardware_failure_does_not_simulate_success(self):
        with patch.object(settings, "motor_driver", "gpio"):
            with patch("app.motors.gpio.GpioMotorDriver", side_effect=RuntimeError("GPIO unavailable")):
                with self.assertRaisesRegex(RuntimeError, "GPIO unavailable"):
                    DriveService().start()


if __name__ == "__main__":
    unittest.main()
