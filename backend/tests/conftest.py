"""Test environment: isolated temp database/media, sim motors, no cloud keys.

Environment variables must be set before ``app.config`` is first imported, so
this happens at conftest import time (pytest loads conftest before any test
module).
"""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="patch-tests-"))

os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["MEDIA_ROOT"] = (_TMP / "media").as_posix()
os.environ["MOTOR_DRIVER"] = "sim"
os.environ["OMNI_API_KEY"] = ""          # never call the real gateway from tests
os.environ["ROVER_SIMULATION"] = "false"
os.environ["DRIVE_MANUAL_TTL_SECONDS"] = "2.0"
# Synthetic frames intentionally have flat fills, unlike a textured camera image.
os.environ["BLUR_VARIANCE_THRESHOLD"] = "0"

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _database():
    from app.db import init_db

    init_db()
    yield


@pytest.fixture
def media_root() -> Path:
    root = _TMP / "media"
    root.mkdir(parents=True, exist_ok=True)
    return root
