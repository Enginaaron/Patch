import os

# Must run before torch/OpenCV are imported anywhere (transitively, via
# app.services.yolo_detector below) -- these libraries read their thread
# count from the environment at import/init time. Calling
# torch.set_num_threads() etc. *after* import was tried first and measured
# live to have no effect (still ~750% CPU / ~7.5 cores for local YOLO
# polling alone); setting the env vars before the first import is what
# actually constrained it.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

# Thread *count* wasn't the actual problem -- reducing the YOLO poll rate
# from 10Hz to 4Hz made zero measurable difference to CPU usage, which ruled
# out "too many calls." Working theory: OpenMP/MKL worker threads spin-wait
# (busy-poll) for a grace period after finishing work before actually
# sleeping, to avoid wake-up latency for the next call. At any poll rate
# with gaps shorter than that grace period, the pool never truly goes idle
# and burns CPU the entire time between calls, not just during them. Forcing
# an immediate, passive sleep instead of spin-waiting is the standard fix.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import init_db
from app.routers.chat import router as chat_router
from app.routers.memory import router as memory_router
from app.routers.control import router as control_router
from app.routers.searches import router as searches_router
from app.routers.speech import router as speech_router
from app.rover.controller import get_rover_controller, set_rover_controller
from app.services.bbox import box_2d_to_pixels
from app.services.camera_service import camera_service
from app.services.drive_service import drive_service
from app.services.yolo_detector import get_tracking_box

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _startup() -> None:
    """Bring the services up. Nothing moves here: no search is resumed by
    itself. A search that was mid-mission when the backend last stopped is
    reported as stopped / interrupted (``RoverController.state_for``) and only
    continues when the user presses resume."""
    init_db()
    if settings.rover_simulation:
        # Laptop demo: a synthetic camera + simulated detections. Refuses
        # (RuntimeError -> the server does not start) unless MOTOR_DRIVER=sim:
        # simulated detections must never reach real motors. Must happen
        # before the camera starts, because the source cannot be swapped while
        # the capture thread runs.
        from app.rover.simulation import install_simulation

        install_simulation(camera_service)
    camera_service.start()
    drive_service.start()
    # Build the controller now rather than on the first request: a
    # misconfiguration fails at startup, and the first /api/rover/stop never
    # pays for construction. Building it issues no motor command.
    get_rover_controller()


def _shutdown(*, controller_started: bool = True) -> None:
    """Every step is attempted even when an earlier one fails, so the motors
    are always released: mission first (so nothing issues another pulse),
    then the motor driver, then the camera."""
    steps = [("drive service", drive_service.close), ("camera", camera_service.stop)]
    if controller_started:
        steps.insert(0, ("rover controller", lambda: get_rover_controller().shutdown()))
    for name, step in steps:
        try:
            step()
        except Exception:  # noqa: BLE001 - keep going: the remaining steps still have to run
            logger.exception("shutdown: %s did not stop cleanly", name)
    # A shut-down controller refuses new missions; forget it so that a later
    # startup in the same process (tests, reload) builds a fresh one.
    set_rover_controller(None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _startup()
    except Exception:
        # Release whatever did come up (camera thread, motor driver) before
        # refusing to start.
        _shutdown(controller_started=False)
        raise
    try:
        yield
    finally:
        _shutdown()


app = FastAPI(title="Patch", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(searches_router)
app.include_router(chat_router)
app.include_router(control_router)
app.include_router(speech_router)
app.include_router(memory_router)

Path(settings.media_root).mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=settings.media_root), name="media")


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


def _draw_tracking_box(frame, box_2d: list[int]):
    """Burns the YOLO-detected box (Spec 11) into the stream frame.

    Styled deliberately unlike a hypothetical OMNI-confirmed candidate box
    (green, "confirmed") -- this cyan "tracking" box means "something
    detected locally," not "OMNI confirmed this is yours." Those are
    different claims and must not look the same on screen.
    """
    height, width = frame.shape[:2]
    box = box_2d_to_pixels(box_2d, width, height)
    frame = frame.copy()
    color = (255, 255, 0)  # cyan (BGR)
    cv2.rectangle(frame, (box.left, box.top), (box.right, box.bottom), color, 2)
    label_y = max(box.top - 8, 12)
    cv2.putText(frame, "tracking", (box.left, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def _mjpeg_frames() -> Iterator[bytes]:
    while True:
        frame = camera_service.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        tracking_box = get_tracking_box()
        if tracking_box is not None:
            frame = _draw_tracking_box(frame, tracking_box)

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


# A built frontend can be served by the Pi without installing Node there.
# Register last so API, camera, and media routes retain priority.
_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@app.get("/assets/{path:path}", include_in_schema=False)
def frontend_asset(path: str):
    path = f"assets/{path}"
    requested = (_frontend_dist / path).resolve()
    if requested.is_relative_to(_frontend_dist.resolve()) and requested.is_file():
        return FileResponse(requested)
    return JSONResponse(status_code=404, content={"detail": "Asset not found"})


@app.get("/", include_in_schema=False)
@app.get("/memory", include_in_schema=False)
@app.get("/search/{path}", include_in_schema=False)
@app.get("/items/{path}", include_in_schema=False)
@app.get("/dev/{path}", include_in_schema=False)
def frontend(path: str = ""):
    index = _frontend_dist / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse(status_code=404, content={"detail": "Frontend not built; run npm run build in frontend"})
