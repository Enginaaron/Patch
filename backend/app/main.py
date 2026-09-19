import time
from contextlib import asynccontextmanager
from typing import Iterator

import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.db import init_db
from app.services.camera_service import camera_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    camera_service.start()
    yield
    camera_service.stop()


app = FastAPI(title="Patch", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/debug/camera")
def debug_camera() -> dict:
    if not camera_service.is_available():
        return {"opened": False, "error": camera_service.error()}

    frame = camera_service.get_frame()
    if frame is None:
        return {"opened": True, "frame_captured": False}

    height, width = frame.shape[:2]
    channels = frame.shape[2] if frame.ndim == 3 else 1
    return {
        "opened": True,
        "frame_captured": True,
        "width": width,
        "height": height,
        "channels": channels,
    }


def _mjpeg_frames() -> Iterator[bytes]:
    while True:
        frame = camera_service.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        jpeg_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n" + jpeg_bytes + b"\r\n"
        )
        time.sleep(1 / 30)


@app.get("/video")
def video():
    if not camera_service.is_available():
        return JSONResponse(status_code=503, content={"error": camera_service.error() or "camera unavailable"})

    return StreamingResponse(_mjpeg_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
