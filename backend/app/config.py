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


settings = Settings()
