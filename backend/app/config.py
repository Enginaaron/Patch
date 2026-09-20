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


settings = Settings()
