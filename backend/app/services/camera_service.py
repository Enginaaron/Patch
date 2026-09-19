import logging
import threading
import time

import numpy as np

from app.camera.webcam import WebcamCamera
from app.config import settings

logger = logging.getLogger(__name__)


class CameraService:
    def __init__(self) -> None:
        self._camera = WebcamCamera(settings.camera_index)
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._available = False
        self._error: str | None = None

    def start(self) -> None:
        try:
            self._camera.start()
        except RuntimeError as exc:
            self._available = False
            self._error = str(exc)
            logger.warning("camera unavailable: %s", exc)
            return

        self._available = True
        self._error = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            frame = self._camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            with self._lock:
                self._latest_frame = frame

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def is_available(self) -> bool:
        return self._available

    def error(self) -> str | None:
        return self._error

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._camera.stop()
        self._available = False


camera_service = CameraService()
