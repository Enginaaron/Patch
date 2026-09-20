"""Tuning for the rover controller.

``RoverConfig.from_settings()`` mirrors the ``rover_*`` fields in
``app.config.Settings``; tests build a ``RoverConfig`` directly with tiny
durations so whole missions run in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings

# Approach tuning. Error is -1..1 across the image: 0.24 = 12% of frame width.
# Increase speed for faster travel; reduce it first if the rover overshoots.
APPROACH_V_MAX = 0.65
# Proportional differential wheel speed; reduce for left/right oscillation.
STEER_KP = 0.70
STEER_MAX = 0.35                 # Raise for stronger turns; capped by wheel limit.
STEER_DEADZONE_ENTER = 0.24      # Enter straight driving within +/-12% of frame width.
STEER_DEADZONE_EXIT = 0.30       # Steer again beyond +/-15%; widen to reduce chatter.
STEER_PIVOT_ENTER = 0.55         # Only pivot for large error; raise to prefer arcs.
STEER_PIVOT_EXIT = 0.50          # Resume arcs below this; lower for more pivot hysteresis.
STEER_ERROR_SLOW = 0.90         # Larger = retain more forward speed while steering.
STEER_FILTER_ALPHA = 0.55       # Lower = smoother but slower response; 1 disables filtering.
APPROACH_MIN_MOTOR = 0.30       # Raise until BOTH wheels start reliably under load.
APPROACH_WHEEL_MAX = 0.75       # Absolute requested wheel cap; lower to soften arcs.
APPROACH_SLOW_SPEED = 0.35      # Forward speed near arrival; keep >= minimum effective PWM.
ARC_TURN_FRACTION = 0.80        # Keep both wheels forward in an arc; raise for tighter arcs.
ARRIVAL_ENTER_SCALE = 0.80      # Multiply category size thresholds; lower stops farther away.
ARRIVAL_EXIT_SCALE = 0.70       # Must drop below this to leave arrival zone; keep < enter.
# One-point calibration from the user's physical measurement and saved track.
# Valid for this bottle/camera setup; different bottle widths change the estimate.
BOTTLE_REFERENCE_DISTANCE_FT = 7.5
BOTTLE_REFERENCE_WIDTH = 0.089
BOTTLE_STOP_DISTANCE_FT = 2.0   # Increase to stop farther away; decrease to get closer.
BOTTLE_ARRIVAL_WIDTH = (BOTTLE_REFERENCE_WIDTH * BOTTLE_REFERENCE_DISTANCE_FT
                        / BOTTLE_STOP_DISTANCE_FT / ARRIVAL_ENTER_SCALE)
APPROACH_STEERING_PULSE_SECONDS = 0.25 # Keep turns/arcs short even when straight travel is longer.
LOST_TARGET_HOLD_SECONDS = 0.65 # Maximum in-flight command hold without a new observation.
LOST_TARGET_TIMEOUT_SECONDS = 2.0 # Stationary retry window after a miss; lower gives up sooner.
# The synthetic camera uses 62 degrees FOV and 8 cm bottles, unlike this USB camera.
SIMULATION_PROXIMITY_PROFILES_JSON = '{"bottle": {"arrive_width": 0.20, "require_both": true}}'


@dataclass
class RoverConfig:
    approach_v_max: float = APPROACH_V_MAX
    steer_kp: float = STEER_KP
    steer_max: float = STEER_MAX
    steer_deadzone_enter: float = STEER_DEADZONE_ENTER
    steer_deadzone_exit: float = STEER_DEADZONE_EXIT
    steer_pivot_enter: float = STEER_PIVOT_ENTER
    steer_pivot_exit: float = STEER_PIVOT_EXIT
    steer_error_slow: float = STEER_ERROR_SLOW
    steer_filter_alpha: float = STEER_FILTER_ALPHA
    approach_min_motor: float = APPROACH_MIN_MOTOR
    approach_wheel_max: float = APPROACH_WHEEL_MAX
    approach_slow_speed: float = APPROACH_SLOW_SPEED
    arc_turn_fraction: float = ARC_TURN_FRACTION
    arrival_enter_scale: float = ARRIVAL_ENTER_SCALE
    arrival_exit_scale: float = ARRIVAL_EXIT_SCALE
    lost_target_hold_seconds: float = LOST_TARGET_HOLD_SECONDS
    lost_target_timeout_seconds: float = LOST_TARGET_TIMEOUT_SECONDS
    approach_steering_pulse_seconds: float = APPROACH_STEERING_PULSE_SECONDS
    # scanning
    turn_speed: float = 0.4
    turn_degrees_per_second: float = 85.0   # approximate, no encoders -- calibrate on the floor
    scan_pulse_seconds: float = 0.35
    scan_max_steps: int = 14
    scan_direction: str = "right"           # "left" | "right"
    settle_seconds: float = 0.4
    frame_timeout_seconds: float = 2.0
    max_frame_failures: int = 3
    omni_check_every_steps: int = 3
    max_candidates: int = 5
    max_vision_errors: int = 3
    inference_timeout_seconds: float = 20.0
    camera_hfov_degrees: float = 62.0
    reject_bearing_tolerance_degrees: float = 15.0
    candidate_likelihood_threshold: str = "high"

    # centring + approach
    turn_pulse_min_seconds: float = 0.08
    turn_pulse_max_seconds: float = 0.30
    center_tolerance: float = 0.08
    forward_speed: float = 0.45
    forward_pulse_seconds: float = 0.65
    slow_forward_speed: float = 0.35
    slow_forward_pulse_seconds: float = 0.20
    slow_zone_fraction: float = 0.7
    arrival_confirmations: int = 2
    approach_min_likelihood: str = "medium"
    approach_max_seconds: float = 90.0
    approach_max_pulses: int = 40
    stall_pulses: int = 3
    min_progress_fraction: float = 0.03
    reacquire_attempts: int = 3
    omni_revalidate_every_pulses: int = 3
    association_max_center_shift: float = 0.30
    association_max_size_ratio: float = 2.0
    pulse_watchdog_margin_seconds: float = 0.15
    proximity_profiles_json: str = ""

    # Whether the local (YOLO) screening layer may be used at all.
    local_detection_enabled: bool = True

    @classmethod
    def from_settings(cls) -> "RoverConfig":
        return cls(
            turn_speed=settings.rover_turn_speed,
            turn_degrees_per_second=settings.rover_turn_degrees_per_second,
            scan_pulse_seconds=settings.rover_scan_pulse_seconds,
            scan_max_steps=settings.rover_scan_max_steps,
            scan_direction=settings.rover_scan_direction,
            settle_seconds=settings.rover_settle_seconds,
            frame_timeout_seconds=settings.rover_frame_timeout_seconds,
            max_frame_failures=settings.rover_max_frame_failures,
            omni_check_every_steps=settings.rover_omni_check_every_steps,
            max_candidates=settings.rover_max_candidates,
            max_vision_errors=settings.rover_max_vision_errors,
            inference_timeout_seconds=settings.rover_inference_timeout_seconds,
            camera_hfov_degrees=settings.rover_camera_hfov_degrees,
            reject_bearing_tolerance_degrees=settings.rover_reject_bearing_tolerance_degrees,
            candidate_likelihood_threshold=settings.candidate_likelihood_threshold,
            turn_pulse_min_seconds=settings.rover_turn_pulse_min_seconds,
            turn_pulse_max_seconds=settings.rover_turn_pulse_max_seconds,
            center_tolerance=settings.rover_center_tolerance,
            forward_speed=settings.rover_forward_speed,
            forward_pulse_seconds=settings.rover_forward_pulse_seconds,
            slow_forward_speed=settings.rover_slow_forward_speed,
            slow_forward_pulse_seconds=settings.rover_slow_forward_pulse_seconds,
            slow_zone_fraction=settings.rover_slow_zone_fraction,
            arrival_confirmations=settings.rover_arrival_confirmations,
            approach_min_likelihood=settings.rover_approach_min_likelihood,
            approach_max_seconds=settings.rover_approach_max_seconds,
            approach_max_pulses=settings.rover_approach_max_pulses,
            stall_pulses=settings.rover_stall_pulses,
            min_progress_fraction=settings.rover_min_progress_fraction,
            reacquire_attempts=settings.rover_reacquire_attempts,
            omni_revalidate_every_pulses=settings.rover_omni_revalidate_every_pulses,
            association_max_center_shift=settings.rover_association_max_center_shift,
            association_max_size_ratio=settings.rover_association_max_size_ratio,
            pulse_watchdog_margin_seconds=settings.rover_pulse_watchdog_margin_seconds,
            proximity_profiles_json=(settings.rover_proximity_profiles_json or
                (SIMULATION_PROXIMITY_PROFILES_JSON if settings.rover_simulation else "")),
            local_detection_enabled=settings.local_detection_enabled,
        )
