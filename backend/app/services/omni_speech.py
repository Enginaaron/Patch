import base64
import json
import logging
import time
from dataclasses import dataclass

import httpx
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Confirmed live against the real yibuapi gateway (2026-09-20):
# - Non-streaming chat.completions with modalities=["text","audio"] is accepted
#   but silently returns message.audio = null -- audio_tokens are billed but no
#   audio data comes back. Streaming (stream=True) is required: audio arrives as
#   base64 chunks in choices[0].delta.audio.data across the SSE stream.
# - The returned audio is raw PCM16 mono at 24kHz with no RIFF/WAV container
#   (OpenAI's realtime-audio convention), not a ready-to-play file -- it must be
#   wrapped before it's a valid .wav.
# - "alloy"/"Cherry" (OpenAI/documented Qwen-Omni voice names) are rejected;
#   "Serena" is a voice this gateway actually accepts.
SYNTHESIS_VOICE = "Serena"
PCM_SAMPLE_RATE = 24000
PCM_SAMPLE_WIDTH = 2  # bytes (16-bit)
PCM_CHANNELS = 1


@dataclass
class TranscriptionResult:
    transcript: str
    latency_seconds: float


@dataclass
class SynthesisResult:
    pcm_bytes: bytes
    sample_rate: int
    sample_width: int
    channels: int
    latency_seconds: float


def _client() -> OpenAI:
    return OpenAI(base_url=settings.omni_base_url, api_key=settings.omni_api_key)


def _audio_data_url(audio_bytes: bytes, audio_format: str) -> str:
    mime = f"audio/{audio_format}"
    return f"data:{mime};base64,{base64.b64encode(audio_bytes).decode('ascii')}"


TRANSCRIBE_PROMPT = (
    "Transcribe the spoken audio. Respond with ONLY the transcribed phrase as plain "
    "text, nothing else -- no punctuation commentary, no quotes, no explanation."
)


def transcribe_audio(audio_bytes: bytes, audio_format: str) -> TranscriptionResult:
    client = _client()
    content = [
        {"type": "text", "text": TRANSCRIBE_PROMPT},
        {
            "type": "input_audio",
            "input_audio": {"data": _audio_data_url(audio_bytes, audio_format), "format": audio_format},
        },
    ]

    start = time.monotonic()
    response = client.chat.completions.create(
        model=settings.omni_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=100,
        temperature=0.0,
    )
    latency_seconds = time.monotonic() - start
    logger.info("OMNI transcribe call took %.2fs", latency_seconds)

    raw_text = response.choices[0].message.content or ""
    transcript = raw_text.strip().strip('"').strip("'").rstrip(".").strip().lower()
    return TranscriptionResult(transcript=transcript, latency_seconds=latency_seconds)


SYNTHESIS_MAX_ATTEMPTS = 2


def synthesize_speech(text: str, voice: str = SYNTHESIS_VOICE) -> SynthesisResult:
    """Requests combined text+audio output. Must be streamed -- see module docstring.

    Retries once on a transient empty-audio response -- verified live that the
    endpoint occasionally returns no audio data at all for a request that
    succeeds moments later with an identical payload.
    """
    last_error: Exception | None = None
    for attempt in range(1, SYNTHESIS_MAX_ATTEMPTS + 1):
        try:
            return _synthesize_once(text, voice)
        except ValueError as exc:
            last_error = exc
            logger.warning("synthesis attempt %d/%d failed: %s", attempt, SYNTHESIS_MAX_ATTEMPTS, exc)
    assert last_error is not None
    raise last_error


def _synthesize_once(text: str, voice: str) -> SynthesisResult:
    start = time.monotonic()
    audio_chunks: list[str] = []

    with httpx.stream(
        "POST",
        settings.omni_base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {settings.omni_api_key}", "Content-Type": "application/json"},
        json={
            "model": settings.omni_model,
            "modalities": ["text", "audio"],
            "audio": {"voice": voice, "format": "wav"},
            # Sending the raw text as a plain user message makes the model treat
            # it as something to respond TO conversationally (e.g. "Still there"
            # -> "Yes, I'm still here, how can I help?") instead of reading it
            # aloud. "Say exactly: X" was tried first and is NOT reliable --
            # verified live across repeated runs, it intermittently produced
            # multi-sentence conversational replies even at temperature 0. This
            # "Repeat this text..." phrasing was the one that held up over 6/6
            # runs across two different phrases (checked by actual audio
            # duration, not just the text echo, since those can diverge).
            "messages": [
                {"role": "user", "content": f"Repeat this text with no changes and no extra words: {text}"}
            ],
            "temperature": 0.0,
            "stream": True,
        },
        timeout=60,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            audio_delta = delta.get("audio")
            if isinstance(audio_delta, dict) and audio_delta.get("data"):
                audio_chunks.append(audio_delta["data"])

    latency_seconds = time.monotonic() - start
    logger.info("OMNI synthesize call took %.2fs", latency_seconds)

    if not audio_chunks:
        raise ValueError("OMNI response contained no audio data")

    pcm_bytes = base64.b64decode("".join(audio_chunks))
    return SynthesisResult(
        pcm_bytes=pcm_bytes,
        sample_rate=PCM_SAMPLE_RATE,
        sample_width=PCM_SAMPLE_WIDTH,
        channels=PCM_CHANNELS,
        latency_seconds=latency_seconds,
    )
