import wave
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.config import settings
from app.services.omni_speech import synthesize_speech, transcribe_audio

router = APIRouter(prefix="/api/speech")


class TranscribeResponse(BaseModel):
    transcript: str
    latency_seconds: float


class SynthesizeRequest(BaseModel):
    text: str


class SynthesizeResponse(BaseModel):
    audio_url: str
    latency_seconds: float


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)) -> TranscribeResponse:
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio file is empty")

    audio_format = Path(audio.filename or "").suffix.lstrip(".").lower() or "webm"

    try:
        # Off the event loop: transcribe_audio() is a blocking network call that
        # takes seconds, and the rover's stop endpoints (/api/rover/stop,
        # /api/drive, /api/drive/stop, cancel) run inline on this loop. Called
        # directly here, every stop would wait for the transcription to finish.
        result = await run_in_threadpool(transcribe_audio, audio_bytes, audio_format)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"transcription failed: {exc}") from exc

    return TranscribeResponse(transcript=result.transcript, latency_seconds=result.latency_seconds)


@router.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(body: SynthesizeRequest) -> SynthesizeResponse:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        result = synthesize_speech(text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"synthesis failed: {exc}") from exc

    speech_dir = Path(settings.media_root) / "speech"
    speech_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4()}.wav"

    with wave.open(str(speech_dir / filename), "wb") as w:
        w.setnchannels(result.channels)
        w.setsampwidth(result.sample_width)
        w.setframerate(result.sample_rate)
        w.writeframes(result.pcm_bytes)

    return SynthesizeResponse(audio_url=f"/media/speech/{filename}", latency_seconds=result.latency_seconds)
