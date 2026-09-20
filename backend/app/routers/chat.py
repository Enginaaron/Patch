import cv2
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.camera_service import camera_service
from app.services.omni_service import ChatMessage, omni_service

router = APIRouter(prefix="/api")


class ChatMessageIn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessageIn]
    target_text: str | None = None


class ChatResponse(BaseModel):
    reply: str
    saw: list[str]
    mode: str


def _current_frame_jpeg() -> bytes | None:
    frame = camera_service.get_frame()
    if frame is None:
        return None
    ok, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if ok else None


@router.get("/omni/status")
def omni_status() -> dict:
    return {"enabled": omni_service.enabled, "mode": omni_service.mode}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    result = await omni_service.generate_reply(
        target_text=req.target_text,
        messages=[ChatMessage(role=m.role, content=m.content) for m in req.messages],
        frame_jpeg=_current_frame_jpeg(),
    )
    return ChatResponse(reply=result.reply, saw=result.saw, mode=result.mode)
