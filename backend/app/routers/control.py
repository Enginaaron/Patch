from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.drive_service import drive_service
from app.services.search_service import search_service

router = APIRouter(prefix="/api")


class DriveCommand(BaseModel):
    command: str
    speed: float | None = None


class StartSearch(BaseModel):
    target_text: str


# --- manual teleop ---------------------------------------------------------


@router.post("/drive")
def drive(cmd: DriveCommand) -> dict:
    # Teleop takes over the wheels, so cancel any autonomous search first.
    search_service.stop()
    try:
        drive_service.drive(cmd.command, cmd.speed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return drive_service.state()


@router.post("/drive/stop")
def drive_stop() -> dict:
    search_service.stop()
    drive_service.stop()
    return drive_service.state()


@router.get("/drive/state")
def drive_state() -> dict:
    return drive_service.state()


# --- autonomous search -----------------------------------------------------


@router.post("/searches/{search_id}/start-auto")
def start_auto(search_id: str, body: StartSearch) -> dict:
    target = body.target_text.strip()
    if not target:
        raise HTTPException(status_code=400, detail="target_text must not be empty")
    return search_service.start(search_id, target)


@router.post("/searches/stop-auto")
def stop_auto() -> dict:
    return search_service.stop()


@router.get("/searches/auto-status")
def auto_status() -> dict:
    return search_service.status()
