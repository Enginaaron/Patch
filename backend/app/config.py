from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    patch_env: str = "development"
    camera_index: int = 0
    gemini_api_key: str = ""
    database_url: str = "sqlite:///./data/patch.db"
    media_root: str = "./data"
    ai_sample_interval_seconds: float = 1.25

    # OMNI (OpenAI-compatible multimodal gateway). Used by omni_vision (search
    # detection), omni_speech (voice) and omni_service (chat). The base URL and
    # model below are the ones verified live on aa_dev; chat path / timeout are
    # configurable so the chat endpoint can be re-pointed via .env.
    omni_api_key: str = ""
    omni_base_url: str = "https://yibuapi.com/v1"
    omni_model: str = "qwen3.5-omni-plus"
    omni_chat_path: str = "/chat/completions"
    omni_timeout_seconds: float = 30.0

    # "low" | "medium" | "high" -- minimum SearchDetection.likelihood required
    # to create a Candidate. Config value so it can be tuned later without a
    # code change.
    candidate_likelihood_threshold: str = "high"

    local_detection_enabled: bool = True
    local_detection_cooldown_seconds: float = 1.25

    # ElevenLabs voice (speech out) — wired later, kept here so the key has a home.
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    # Drive / motors. "sim" logs intended wheel speeds and needs no hardware;
    # "gpio" drives the real TB6612FNG differential base.
    motor_driver: str = "sim"  # "sim" | "gpio"
    drive_max_speed: float = 1.0  # ceiling applied to every command, 0..1
    # Per-wheel PWM calibration; 1 keeps the requested power unchanged.
    motor_left_scale: float = Field(default=1.0, gt=0, le=2, allow_inf_nan=False)
    motor_right_scale: float = Field(default=1.0, gt=0, le=2, allow_inf_nan=False)

    # TB6612FNG pins (BCM numbering), gpio driver only. Each motor uses two
    # direction pins (IN1/IN2) plus a PWM pin; STBY enables the whole chip.
    motor_standby_pin: int = 16
    motor_left_pwm_pin: int = 12   # PWMA (hardware-PWM capable)
    motor_left_in1_pin: int = 5   # AIN1
    motor_left_in2_pin: int = 6   # AIN2
    motor_right_pwm_pin: int = 13  # PWMB (hardware-PWM capable)
    motor_right_in1_pin: int = 20  # BIN1
    motor_right_in2_pin: int = 21  # BIN2
    motor_pwm_hz: int = 1000
    # If a wheel spins the wrong way, flip its invert flag instead of rewiring.
    motor_left_invert: bool = False
    motor_right_invert: bool = False

    # Deadman for manual /api/drive commands: wheels auto-stop this many
    # seconds after the last command unless it is re-sent (the ControlPad
    # re-sends while a button is held). 0 disables.
    drive_manual_ttl_seconds: float = 2.0

    # Camera: a frame older than this is treated as stale (get_frame() -> None).
    camera_stale_after_seconds: float = 2.0

    # OMNI vision call bounds (omni_vision). The rover controller additionally
    # enforces rover_inference_timeout_seconds on every call it makes.
    omni_vision_timeout_seconds: float = 20.0
    omni_vision_max_retries: int = 0

    # --- Rover autonomy (app/rover). See docs/AUTONOMY.md ------------------
    # ROVER_SIMULATION=true swaps in a synthetic camera + vision world for the
    # laptop demo. It is refused unless MOTOR_DRIVER=sim: simulated detections
    # must never move real motors.
    rover_simulation: bool = False
    rover_sim_scenario: str = "two_bottles"

    # Scanning: rotate-in-place pulses with a stop + settle + fresh frame
    # between each. Angles are approximate (no encoders) -- calibrate
    # rover_turn_degrees_per_second at rover_turn_speed (docs/HARDWARE.md).
    rover_turn_speed: float = 0.4
    rover_turn_degrees_per_second: float = 85.0
    rover_scan_pulse_seconds: float = 0.35
    rover_scan_max_steps: int = 14          # bounded scan budget (~1 rotation + margin)
    rover_scan_direction: str = "right"     # "left" | "right"
    rover_settle_seconds: float = 0.4       # wait after the wheels stop before trusting a frame
    rover_frame_timeout_seconds: float = 2.0
    rover_max_frame_failures: int = 3
    rover_omni_check_every_steps: int = 3   # periodic OMNI check even when YOLO sees nothing
    rover_max_candidates: int = 5           # candidate questions per search before giving up
    rover_max_vision_errors: int = 3        # consecutive inference failures before phase=error
    rover_inference_timeout_seconds: float = 20.0
    rover_camera_hfov_degrees: float = 62.0
    rover_reject_bearing_tolerance_degrees: float = 15.0

    # Centring + approach (short pulses, observe between each).
    rover_turn_pulse_min_seconds: float = 0.08
    rover_turn_pulse_max_seconds: float = 0.30
    rover_center_tolerance: float = 0.08    # |box centre - 0.5| under this counts as centred
    rover_forward_speed: float = 0.45
    rover_forward_pulse_seconds: float = 0.40
    rover_slow_forward_speed: float = 0.35
    rover_slow_forward_pulse_seconds: float = 0.20
    rover_slow_zone_fraction: float = 0.7   # slow down once size >= this fraction of the arrival threshold
    rover_arrival_confirmations: int = 2    # consecutive stationary observations needed to declare arrival
    rover_approach_min_likelihood: str = "medium"
    rover_approach_max_seconds: float = 90.0
    rover_approach_max_pulses: int = 40
    rover_stall_pulses: int = 3             # pulses without useful progress before stopping
    rover_min_progress_fraction: float = 0.03
    rover_reacquire_attempts: int = 3       # stationary re-observations before declaring the target lost
    rover_omni_revalidate_every_pulses: int = 3
    rover_association_max_center_shift: float = 0.30
    rover_association_max_size_ratio: float = 2.0
    rover_pulse_watchdog_margin_seconds: float = 0.15
    # JSON object overriding/extending the per-category proximity thresholds in
    # app/rover/proximity.py, e.g. {"bottle": {"arrive_width": 0.2}}
    rover_proximity_profiles_json: str = ""


settings = Settings()
