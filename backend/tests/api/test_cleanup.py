"""The old autonomous loop and its fabricated detections are gone, and the
configuration files keep up with the settings."""

import asyncio
import importlib
import re
from pathlib import Path

import pytest

from app.config import Settings
from app.services import omni_service as omni_module
from app.services.omni_service import ChatMessage, OmniLiveService

BACKEND = Path(__file__).resolve().parents[2]

# Every setting added for the rover work (spec section 4) must be documented in .env.example.
_NEW_SETTINGS = {"drive_manual_ttl_seconds", "camera_stale_after_seconds", "omni_vision_timeout_seconds", "omni_vision_max_retries"}


def test_the_old_search_loop_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.services.search_service")
    assert not (BACKEND / "app" / "services" / "search_service.py").exists()


def test_no_fabricated_target_locations_exist_near_the_motors():
    for name in ("TargetLocation", "locate_target", "_locate_live", "_locate_demo"):
        assert not hasattr(omni_module, name) and not hasattr(OmniLiveService, name)
    source = (BACKEND / "app" / "services" / "omni_service.py").read_text(encoding="utf-8")
    assert "def locate" not in source and "def _locate" not in source


def test_the_demo_chat_reply_still_works_without_a_key():
    service = OmniLiveService()
    assert service.mode == "demo"                        # OMNI_API_KEY is blank in tests

    reply = asyncio.run(
        service.generate_reply(
            target_text="my bottle", messages=[ChatMessage(role="user", content="can you find my bottle?")], frame_jpeg=None
        )
    )

    assert reply.mode == "demo" and "bottle" in reply.reply   # words only -- nothing here can steer the rover


def _env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (BACKEND / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value
    return values


def test_env_example_documents_every_new_setting_with_its_default():
    example = _env_example()
    wanted = {name for name in Settings.model_fields if name.startswith("rover_")} | _NEW_SETTINGS
    assert wanted <= {key.lower() for key in example}, sorted(wanted - {key.lower() for key in example})

    # No trailing "# comment" on a value line: python-dotenv would strip it, but
    # systemd's EnvironmentFile and docker --env-file would make it part of the value.
    assert [key for key, value in example.items() if re.search(r"\s#", value)] == []

    # Copying the example must not change behaviour: every value is the code default.
    defaults = Settings.model_construct()
    parsed = Settings.model_validate({key.lower(): value for key, value in example.items() if key.lower() in wanted})
    for name in sorted(wanted):
        assert getattr(parsed, name) == getattr(defaults, name), name


def _requirements(filename: str) -> list[str]:
    lines = (BACKEND / filename).read_text(encoding="utf-8").splitlines()
    return [line.strip().lower() for line in lines if line.strip() and not line.strip().startswith("#")]


def test_torch_and_ultralytics_are_not_required_and_pytest_is_a_dev_dependency():
    runtime = _requirements("requirements.txt")
    assert not any(re.match(r"(ultralytics|torch)\b", line) for line in runtime)
    assert {"fastapi", "httpx", "openai", "sqlmodel"} <= set(runtime)
    assert "pytest" not in runtime

    dev = _requirements("requirements-dev.txt")
    assert "-r requirements.txt" in dev and "pytest" in dev
