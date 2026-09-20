from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    patch_env: str = "development"
    camera_index: int = 0
    gemini_api_key: str = ""
    database_url: str = "sqlite:///./data/patch.db"
    media_root: str = "./data"
    ai_sample_interval_seconds: float = 1.25

    # Huawei OMNI Live (multimodal reasoning over the live camera frame).
    # Leave omni_api_key blank to run the fake/demo chat; set it to switch to
    # the real model with no other code changes. Base URL / model / path are
    # configurable so the exact OMNI endpoint can be pointed in via .env once
    # the credentials and docs are confirmed.
    omni_api_key: str = ""
    omni_base_url: str = "https://api.omni-live.huaweicloud.com/v1"
    omni_model: str = "omni-live"
    omni_chat_path: str = "/chat/completions"
    omni_timeout_seconds: float = 30.0

    # ElevenLabs voice (speech out) — wired later, kept here so the key has a home.
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    # Drive / motors. "sim" logs intended wheel speeds and needs no hardware;
    # "gpio" drives the real TB6612FNG differential base.
    motor_driver: str = "sim"  # "sim" | "gpio"
    drive_max_speed: float = 1.0  # ceiling applied to every command, 0..1

    # TB6612FNG pins (BCM numbering), gpio driver only. Each motor uses two
    # direction pins (IN1/IN2) plus a PWM pin; STBY enables the whole chip.
    motor_standby_pin: int = 25
    motor_left_pwm_pin: int = 12   # PWMA (hardware-PWM capable)
    motor_left_in1_pin: int = 17   # AIN1
    motor_left_in2_pin: int = 27   # AIN2
    motor_right_pwm_pin: int = 13  # PWMB (hardware-PWM capable)
    motor_right_in1_pin: int = 23  # BIN1
    motor_right_in2_pin: int = 24  # BIN2
    motor_pwm_hz: int = 1000
    # If a wheel spins the wrong way, flip its invert flag instead of rewiring.
    motor_left_invert: bool = False
    motor_right_invert: bool = False

    # Autonomous search tuning.
    search_interval_seconds: float = 0.6  # how often the loop looks + steers
    search_scan_speed: float = 0.5        # rotate-in-place speed while looking
    search_turn_speed: float = 0.45       # rotate speed while centering the target
    search_forward_speed: float = 0.55    # approach speed once centered
    search_center_tolerance: float = 0.12  # |x-0.5| under this counts as centered
    search_arrival_size: float = 0.45     # target width fraction that means "arrived"
    search_min_confidence: float = 0.35   # ignore detections below this


settings = Settings()
