"""OMNI Live reasoning service for Patch.

Patch talks to the user about what it currently sees through its camera. This
service is the single seam between the app and Huawei's OMNI Live multimodal
model:

* When ``settings.omni_api_key`` is set, ``generate_reply`` sends the live
  camera frame plus the conversation to the real model and returns its answer
  (``mode == "live"``).
* When it is blank, it returns a context-aware *fake* reply so the demo and the
  whole UI work end to end today (``mode == "demo"``).

Nothing else in the app needs to know which mode is active. Drop the key in
``.env`` and the same endpoint starts producing real grounded answers.

This service is CHAT ONLY. It used to also answer "where is the target?" for
the old autonomous loop, with a scripted fake answer when no key was set. That
is gone on purpose: a fabricated detection must never exist anywhere near the
motors. The rover's recognition goes through ``app/rover/vision.py``
(``omni_vision`` + ``yolo_detector``), which raises without a key instead of
inventing a result. The demo replies below only ever produce text.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class OmniReply:
    reply: str
    saw: list[str] = field(default_factory=list)
    mode: str = "demo"  # "live" | "demo"


SYSTEM_PROMPT = (
    "You are Patch, a small physical search-and-assistance rover. You see the "
    "world through a single forward camera; the most recent frame is attached. "
    "Speak naturally and briefly, like a helpful companion, not a command "
    "parser. Ground everything you say in what is actually visible in the "
    "frame. When the user refers to objects vaguely (\"the one next to my "
    "backpack\", \"the red one\"), resolve the reference using the scene. If "
    "you are unsure which object they mean, ask one short clarifying question. "
    "Keep replies to one or two sentences."
)


class OmniLiveService:
    def __init__(self) -> None:
        self._api_key = settings.omni_api_key.strip()

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    @property
    def mode(self) -> str:
        return "live" if self.enabled else "demo"

    async def generate_reply(
        self,
        *,
        target_text: str | None,
        messages: list[ChatMessage],
        frame_jpeg: bytes | None,
    ) -> OmniReply:
        if self.enabled:
            try:
                return await self._generate_live(target_text, messages, frame_jpeg)
            except Exception:  # noqa: BLE001 - never let a live failure break the demo
                logger.exception("OMNI Live request failed; falling back to demo reply")
                fallback = self._generate_demo(target_text, messages, frame_jpeg)
                fallback.reply = f"(OMNI Live unavailable, demo reply) {fallback.reply}"
                return fallback

        return self._generate_demo(target_text, messages, frame_jpeg)

    # --- real model ------------------------------------------------------

    async def _generate_live(
        self,
        target_text: str | None,
        messages: list[ChatMessage],
        frame_jpeg: bytes | None,
    ) -> OmniReply:
        """Call the real OMNI Live multimodal chat endpoint.

        Built against the widely used OpenAI-compatible multimodal chat shape
        (``messages`` with mixed text/image_url content). If the OMNI gateway
        expects a different envelope, this is the ONE place to adjust it —
        ``settings.omni_base_url`` / ``omni_chat_path`` / ``omni_model`` cover
        the common differences without touching code.
        """
        system_content = SYSTEM_PROMPT
        if target_text:
            system_content += f"\n\nThe user is currently searching for: {target_text!r}."

        api_messages: list[dict] = [{"role": "system", "content": system_content}]
        for m in messages:
            api_messages.append({"role": m.role, "content": m.content})

        # Attach the live frame to the latest user turn as an inline image.
        if frame_jpeg and api_messages and api_messages[-1]["role"] == "user":
            b64 = base64.b64encode(frame_jpeg).decode("ascii")
            text = api_messages[-1]["content"]
            api_messages[-1]["content"] = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ]

        url = settings.omni_base_url.rstrip("/") + "/" + settings.omni_chat_path.lstrip("/")
        payload = {
            "model": settings.omni_model,
            "messages": api_messages,
            "max_tokens": 300,
            "temperature": 0.4,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        async with httpx.AsyncClient(timeout=settings.omni_timeout_seconds) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        reply = data["choices"][0]["message"]["content"]
        if isinstance(reply, list):  # some gateways return content parts
            reply = "".join(part.get("text", "") for part in reply)

        return OmniReply(reply=reply.strip(), saw=[], mode="live")

    # --- demo / fake -----------------------------------------------------

    def _generate_demo(
        self,
        target_text: str | None,
        messages: list[ChatMessage],
        frame_jpeg: bytes | None,
    ) -> OmniReply:
        """Deterministic, context-aware fake reply so the UI works with no key."""
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        ).strip()
        text = last_user.lower()
        target = (target_text or self._guess_target(text) or "it").strip()

        # A tiny fake "scene" so replies feel grounded during the demo.
        saw = ["a backpack", "a water bottle", "a coffee mug", "a notebook"]

        def has(*words: str) -> bool:
            return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)

        if not last_user:
            reply = "Hi, I'm Patch. Tell me what to look for and I'll scan the room."
        elif has("hi", "hello", "hey"):
            reply = "Hey! I'm looking around now. What should I find for you?"
        elif has("what", "see", "looking at") and has("see", "there", "around"):
            reply = f"Right now I can see {self._join(saw)}. Want me to go toward one of them?"
        elif has("left"):
            reply = "Turning to my left and scanning as I go."
        elif has("right"):
            reply = "Turning to my right — keeping an eye out."
        elif has("closer", "forward", "approach", "go"):
            reply = f"Moving closer to the {target} now."
        elif has("stop", "wait", "hold"):
            reply = "Stopping here. Just say the word when you're ready."
        elif has("no", "not that", "wrong"):
            reply = "Got it, not that one. Can you tell me a colour or what it's next to?"
        elif has("yes", "that one", "correct"):
            reply = f"Great — heading to the {target}."
        elif has("find", "look for", "search", "where", "locate") or target_text:
            reply = (
                f"Scanning for the {target}. I can see {self._join(saw)} — "
                f"is the {target} near any of those?"
            )
        else:
            reply = (
                f"I think you mean the {target}. I see {self._join(saw)} in front of me — "
                "which one is it closest to?"
            )

        if frame_jpeg is None:
            reply += " (Heads up: my camera feed isn't live right now.)"

        return OmniReply(reply=reply, saw=saw, mode="demo")

    @staticmethod
    def _guess_target(text: str) -> str | None:
        for pat in (r"find (?:my |the |a |an )?([\w ]+)", r"looking for (?:my |the |a |an )?([\w ]+)"):
            m = re.search(pat, text)
            if m:
                return m.group(1).strip().rstrip("?.!")
        return None

    @staticmethod
    def _join(items: list[str]) -> str:
        if len(items) <= 1:
            return items[0] if items else "nothing"
        return ", ".join(items[:-1]) + f" and {items[-1]}"


omni_service = OmniLiveService()
