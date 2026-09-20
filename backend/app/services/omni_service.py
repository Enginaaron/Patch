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
"""

from __future__ import annotations

import base64
import json
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


@dataclass
class TargetLocation:
    """Where the search target is in the current frame.

    ``x`` is the horizontal centre, 0.0 = far left, 1.0 = far right. ``size`` is
    the fraction of the frame width the target spans (a rough distance proxy —
    bigger means closer). ``done`` means Patch is close enough to declare a find.
    """

    found: bool
    x: float = 0.5
    size: float = 0.0
    confidence: float = 0.0
    done: bool = False
    note: str = ""
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

    # --- target localisation (drives the autonomous search loop) ---------

    def locate_target(
        self,
        *,
        target_text: str,
        frame_jpeg: bytes | None,
        tick: int = 0,
    ) -> TargetLocation:
        """Report where ``target_text`` is in the current frame.

        Synchronous so the search loop (a background thread) can call it
        directly. Uses the real model when keyed, otherwise a deterministic
        simulation driven by ``tick`` so the whole search loop is demoable.
        """
        if self.enabled:
            try:
                return self._locate_live(target_text, frame_jpeg)
            except Exception:  # noqa: BLE001
                logger.exception("OMNI Live locate failed; using demo locate")
        return self._locate_demo(target_text, tick)

    def _locate_live(self, target_text: str, frame_jpeg: bytes | None) -> TargetLocation:
        instruction = (
            "You are the vision system of a rover. Find this object in the image: "
            f"{target_text!r}. Reply with ONLY a JSON object, no prose, with keys: "
            "found (bool), x (0..1 horizontal centre of the object, 0=left 1=right), "
            "size (0..1 fraction of image width the object spans), "
            "confidence (0..1), done (bool, true if the object fills much of the "
            "frame and is centred), note (short string)."
        )
        content: list | str = instruction
        if frame_jpeg:
            b64 = base64.b64encode(frame_jpeg).decode("ascii")
            content = [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]

        url = settings.omni_base_url.rstrip("/") + "/" + settings.omni_chat_path.lstrip("/")
        payload = {
            "model": settings.omni_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 200,
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        with httpx.Client(timeout=settings.omni_timeout_seconds) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        raw = data["choices"][0]["message"]["content"]
        if isinstance(raw, list):
            raw = "".join(part.get("text", "") for part in raw)
        parsed = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        return TargetLocation(
            found=bool(parsed.get("found", False)),
            x=float(parsed.get("x", 0.5)),
            size=float(parsed.get("size", 0.0)),
            confidence=float(parsed.get("confidence", 0.0)),
            done=bool(parsed.get("done", False)),
            note=str(parsed.get("note", "")),
            mode="live",
        )

    def _locate_demo(self, target_text: str, tick: int) -> TargetLocation:
        """Simulate a search: scan a few ticks, spot the target off to one side,
        centre it, then close in — so the drive loop visibly does its thing."""
        if tick < 3:
            return TargetLocation(
                found=False, confidence=0.1, note="scanning…", mode="demo"
            )

        progress = tick - 3
        # Enters from the left, converges to centre over a few ticks.
        x = min(0.5, 0.18 + 0.08 * progress)
        size = min(settings.search_arrival_size, 0.14 + 0.05 * progress)
        confidence = min(0.9, 0.55 + 0.05 * progress)
        done = size >= settings.search_arrival_size
        note = "arrived" if done else f"found {target_text}, closing in"
        return TargetLocation(
            found=True, x=x, size=size, confidence=confidence, done=done, note=note, mode="demo"
        )

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
