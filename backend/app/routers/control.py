"""Manual driving, plus the rover-wide stop and state endpoints.

Every handler here that can halt the wheels is ``async def`` and calls only
non-blocking code (the controller's public API never waits on inference, frames,
sleeps or joins; the drive service only takes a short lock). FastAPI runs
``async def`` handlers inline on the event loop, while plain ``def`` handlers
queue for a threadpool that speech synthesis, MJPEG streams and slow database
reads can fill up. A stop must never wait in that queue.

The flip side: nothing in these handlers may block, or it would hold up every
other stop. Keep them that way.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.rover.controller import get_rover_controller
from app.rover.types import StopReason
from app.services.camera_service import camera_service
from app.services.drive_service import drive_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class DriveCommand(BaseModel):
    command: str
    speed: float | None = None


def _manual_ttl() -> float | None:
    """Deadman for manual commands: the wheels stop by themselves unless the
    command is re-sent (the ControlPad does, while a button is held), so a
    dropped connection or a closed tab cannot leave the rover driving."""
    ttl = settings.drive_manual_ttl_seconds
    return ttl if ttl > 0 else None  # 0 (or a nonsense negative) disables it


def _mirror_to_simulation(command: str, speed: float | None, ttl: float | None) -> None:
    """Laptop demo only (ROVER_SIMULATION=true): manual commands go to the
    drive service, not through the controller's TeeDrive, so mirror them into
    the simulated world here or the synthetic camera view would not move.
    Touches the simulation only -- never a motor."""
    if not settings.rover_simulation:
        return
    try:
        from app.rover.simulation import get_sim_world

        get_sim_world().apply_command(command, speed, ttl)
    except Exception:  # noqa: BLE001 - cosmetic; must never get in the way of a drive/stop request
        logger.exception("could not mirror %r into the simulated world", command)


def _halt(reason: StopReason) -> None:
    """Stop everything that can turn a wheel. The controller goes first so the
    mission is invalidated and cannot issue another pulse; the drive service is
    then zeroed directly as well, even if the controller call failed."""
    try:
        get_rover_controller().stop(reason)
    finally:
        drive_service.stop()
        _mirror_to_simulation("stop", None, None)


def _rover_state() -> dict:
    return {
        **get_rover_controller().snapshot(),
        "drive": drive_service.state(),
        "camera": camera_service.health(),
    }


# --- manual teleop ---------------------------------------------------------


@router.post("/drive")
async def drive(cmd: DriveCommand) -> dict:
    # A human on the controls always wins. Invalidate the mission BEFORE the
    # wheels get the manual command: from here on the mission cannot start
    # another pulse, and its end-of-pulse stop will not cut this command short.
    # This runs even when the command below turns out to be invalid -- someone
    # reached for the controls, so the autonomous search stops either way.
    get_rover_controller().manual_override()
    ttl = _manual_ttl()
    try:
        drive_service.drive(cmd.command, cmd.speed, ttl=ttl)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _mirror_to_simulation(cmd.command, settings.drive_max_speed if cmd.speed is None else cmd.speed, ttl)
    return drive_service.state()


@router.post("/drive/stop")
async def drive_stop() -> dict:
    _halt(StopReason.MANUAL_OVERRIDE)
    return drive_service.state()


@router.get("/drive/state")
def drive_state() -> dict:
    return drive_service.state()


# --- rover-wide stop / state -------------------------------------------------


@router.post("/rover/stop")
async def rover_stop() -> dict:
    """E-stop. Halts the rover NOW and invalidates the mission; the search
    itself is untouched, so it can be resumed."""
    _halt(StopReason.USER_STOP)
    return _rover_state()


@router.get("/rover/state")
async def rover_state() -> dict:
    return _rover_state()
