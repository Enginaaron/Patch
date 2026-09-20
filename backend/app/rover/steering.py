"""Stateful approach steering; tuning lives at the top of rover/config.py.

No detection or motor IO here. Each valid observation produces one bounded
wheel command. Positive error turns right (left wheel runs faster).
"""
from dataclasses import dataclass, replace
import math

from app.rover.config import RoverConfig
from app.rover.proximity import ProximityProfile, arrival_evidence


@dataclass(frozen=True)
class WheelCommand:
    left: float
    right: float
    pivot: bool
    error: float

    @property
    def label(self) -> str:
        if self.pivot:
            return "turn_right" if self.left > self.right else "turn_left"
        if abs(self.left - self.right) < 1e-9:
            return "forward"
        return "veer_right" if self.left > self.right else "veer_left"


class ApproachSteering:
    def __init__(self, config: RoverConfig):
        self.cfg = config
        self.error: float | None = None
        self.straight = False
        self.pivot = False
        self.arriving = False
        if not (0 <= config.steer_deadzone_enter < config.steer_deadzone_exit
                < config.steer_pivot_exit <= config.steer_pivot_enter < 1):
            raise ValueError("Steering thresholds must be ordered deadzone enter/exit < pivot exit/enter")
        if not (0 < config.approach_min_motor <= config.approach_wheel_max <= 1
                and 0 < config.steer_filter_alpha <= 1
                and 0 < config.arrival_exit_scale < config.arrival_enter_scale <= 1
                and config.steer_error_slow > config.steer_pivot_enter
                and 0 < config.arc_turn_fraction < 1
                and config.steer_kp > 0 and config.steer_max > 0
                and config.approach_v_max > 0 and config.approach_slow_speed > 0
                and config.lost_target_hold_seconds > 0 and config.lost_target_timeout_seconds > 0):
            raise ValueError("Invalid approach speed, filtering, arrival, or timeout tuning")

    def arrival(self, box, profile: ProximityProfile) -> bool:
        scale = self.cfg.arrival_exit_scale if self.arriving else self.cfg.arrival_enter_scale
        threshold = replace(profile, arrive_width=profile.arrive_width * scale,
                            arrive_height=profile.arrive_height * scale)
        self.arriving = arrival_evidence(box, threshold)
        return self.arriving

    def wheels(self, error: float, *, slow: bool = False) -> WheelCommand:
        if not math.isfinite(error):
            raise ValueError("Steering error must be finite")
        cfg = self.cfg
        raw = max(-1.0, min(1.0, error))
        self.error = raw if self.error is None else (
            cfg.steer_filter_alpha * raw + (1 - cfg.steer_filter_alpha) * self.error)
        err = self.error
        # Raw edge excursions immediately suppress forward motion. Filtering
        # must not keep driving toward an object that just left the centre.
        if abs(raw) > cfg.steer_pivot_enter:
            self.pivot = True
        elif abs(raw) <= cfg.steer_pivot_exit:
            self.pivot = False
        if self.straight:
            self.straight = abs(err) <= cfg.steer_deadzone_exit
        else:
            self.straight = abs(err) <= cfg.steer_deadzone_enter
        v_max = cfg.approach_slow_speed if slow else cfg.approach_v_max
        if self.pivot:
            turn = math.copysign(min(cfg.steer_max, cfg.steer_kp * abs(raw)), raw)
            v = 0.0
        elif self.straight:
            v, turn = v_max, 0.0
        else:
            v = v_max * max(0.0, 1 - abs(err) / cfg.steer_error_slow)
            v = max(cfg.approach_min_motor, v)
            turn = max(-cfg.steer_max, min(cfg.steer_max, cfg.steer_kp * err))
            turn = max(-v * cfg.arc_turn_fraction, min(v * cfg.arc_turn_fraction, turn))
        left, right = v + turn, v - turn
        scale = max(1.0, max(abs(left), abs(right)) / cfg.approach_wheel_max)

        def effective(value):
            value /= scale
            return 0.0 if value == 0 else math.copysign(max(cfg.approach_min_motor, abs(value)), value)

        return WheelCommand(effective(left), effective(right), self.pivot, err)
