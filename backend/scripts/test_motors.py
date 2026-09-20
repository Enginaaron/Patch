"""Motor wiring / direction check.

Run on the Pi AFTER wiring the TB6612 and setting MOTOR_DRIVER=gpio in .env:

    cd backend
    source .venv/bin/activate
    python -m scripts.test_motors

It drives each wheel forward then backward, then both together. Watch the
wheels and confirm:

  * "LEFT forward" spins the left wheel so the rover would go forward.
  * "RIGHT forward" does the same on the right.

If a wheel goes the wrong way, set MOTOR_LEFT_INVERT / MOTOR_RIGHT_INVERT=true
in .env (no rewiring needed). If the WHEELS are swapped (left cmd moves the
right wheel), swap the A/B motor output wires on the TB6612.
"""

import time

from app.services.drive_service import drive_service


def main() -> None:
    drive_service.start()
    print(f"driver = {drive_service.state()['driver']}")
    steps = [
        ("LEFT forward", (0.6, 0.0)),
        ("LEFT backward", (-0.6, 0.0)),
        ("RIGHT forward", (0.0, 0.6)),
        ("RIGHT backward", (0.0, -0.6)),
        ("BOTH forward", (0.6, 0.6)),
        ("ROTATE in place", (-0.6, 0.6)),
    ]
    try:
        for label, (l, r) in steps:
            print(label)
            drive_service.set_speeds(l, r, command="test")
            time.sleep(1.5)
            drive_service.stop()
            time.sleep(0.6)
    finally:
        drive_service.close()
        print("done")


if __name__ == "__main__":
    main()
