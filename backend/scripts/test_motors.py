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

import argparse
import math
import time

from app.config import settings
from app.services.drive_service import COMMANDS, drive_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Patch's two front gearmotors. Lift the wheels first.")
    parser.add_argument("--sim", action="store_true", help="Log commands without using GPIO")
    parser.add_argument("--command", choices=["sequence", *COMMANDS], default="sequence")
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--left-scale", type=float, help="Temporary left power multiplier (0 < value <= 2)")
    parser.add_argument("--right-scale", type=float, help="Temporary right power multiplier (0 < value <= 2)")
    parser.add_argument("--duration", type=float, default=1.0, help="Seconds per movement (maximum 5)")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error("speed must be greater than 0 and at most 1")
    if not math.isfinite(args.duration) or not 0 < args.duration <= 5:
        parser.error("duration must be greater than 0 and at most 5 seconds")
    for side in ("left", "right"):
        scale = getattr(args, f"{side}_scale")
        if scale is not None:
            if not math.isfinite(scale) or not 0 < scale <= 2:
                parser.error(f"{side}-scale must be greater than 0 and at most 2")
            setattr(settings, f"motor_{side}_scale", scale)
    settings.motor_driver = "sim" if args.sim else "gpio"
    drive_service.start()
    print(f"driver = {drive_service.state()['driver']}")
    print(f"Power scales: left={settings.motor_left_scale:g}, right={settings.motor_right_scale:g}")
    steps = [
        ("LEFT forward", (1.0, 0.0)),
        ("LEFT backward", (-1.0, 0.0)),
        ("RIGHT forward", (0.0, 1.0)),
        ("RIGHT backward", (0.0, -1.0)),
        ("BOTH forward", (1.0, 1.0)),
        ("BOTH backward", (-1.0, -1.0)),
        ("TURN left", (-1.0, 1.0)),
        ("TURN right", (1.0, -1.0)),
    ]
    if args.command != "sequence":
        steps = [(args.command, COMMANDS[args.command])]
    try:
        for label, (l, r) in steps:
            print(label)
            drive_service.set_speeds(l * args.speed, r * args.speed, command="test")
            state = drive_service.state()
            print(f"  Applied PWM: left={state['left']:+.2f}, right={state['right']:+.2f}", flush=True)
            time.sleep(args.duration)
            drive_service.stop()
            time.sleep(0.6)
    except KeyboardInterrupt:
        print("Interrupted; stopping motors.")
    finally:
        drive_service.close()
        print("done")


if __name__ == "__main__":
    main()
