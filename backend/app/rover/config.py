"""Tuning for the rover controller.

``RoverConfig.from_settings()`` mirrors the ``rover_*`` fields in
``app.config.Settings``; tests build a ``RoverConfig`` directly with tiny
durations so whole missions run in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings


@dataclass
class RoverConfig:
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
    forward_pulse_seconds: float = 0.40
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
            proximity_profiles_json=settings.rover_proximity_profiles_json,
            local_detection_enabled=settings.local_detection_enabled,
        )
