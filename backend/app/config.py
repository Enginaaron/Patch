from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    patch_env: str = "development"
    camera_index: int = 0
    gemini_api_key: str = ""
    database_url: str = "sqlite:///./data/patch.db"
    media_root: str = "./data"
    ai_sample_interval_seconds: float = 1.25

    omni_api_key: str = ""
    omni_base_url: str = "https://yibuapi.com/v1"
    omni_model: str = "qwen3.5-omni-plus"

    # "low" | "medium" | "high" -- minimum SearchDetection.likelihood required
    # to create a Candidate. Config value so it can be tuned later without a
    # code change.
    candidate_likelihood_threshold: str = "high"

    local_detection_enabled: bool = True
    local_detection_cooldown_seconds: float = 1.25


settings = Settings()
