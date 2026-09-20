import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from openai import OpenAI
from PIL import ImageOps
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, model_validator

from app.config import settings

if TYPE_CHECKING:
    # Type-only: this module stays importable without the rover package, and
    # build_content() just reads .description / .crop_jpeg off each hint.
    from app.rover.types import RejectedHint

logger = logging.getLogger(__name__)


class SearchDetection(BaseModel):
    found: bool
    likelihood: Literal["low", "medium", "high"]
    box_2d: list[int] | None = None
    description: str

    @model_validator(mode="after")
    def _validate_box(self) -> "SearchDetection":
        if not self.found:
            if self.box_2d is not None:
                raise ValueError("box_2d must be null when found is false")
            return self

        if self.box_2d is not None:
            if len(self.box_2d) != 4:
                raise ValueError("box_2d must have exactly 4 integers [ymin, xmin, ymax, xmax]")
            if not all(isinstance(v, int) for v in self.box_2d):
                raise ValueError("box_2d values must be integers")
            if not all(0 <= v <= 1000 for v in self.box_2d):
                raise ValueError("box_2d values must be in [0, 1000]")
            ymin, xmin, ymax, xmax = self.box_2d
            if ymin >= ymax:
                raise ValueError("box_2d ymin must be < ymax")
            if xmin >= xmax:
                raise ValueError("box_2d xmin must be < xmax")
        return self


@dataclass
class DetectionResult:
    detection: SearchDetection
    latency_seconds: float


class _ModelDetection(BaseModel):
    """Provider coordinates are XYXY; the application's Box remains YXYX."""

    model_config = ConfigDict(extra="forbid")
    found: StrictBool
    likelihood: Literal["low", "medium", "high"]
    bbox_2d: list[StrictInt] | None
    description: str

    @model_validator(mode="before")
    @classmethod
    def normalize_absent_target(cls, data):
        # The gateway sometimes omits metadata for an explicit negative result.
        # Only that unambiguous case can authorize continuing the scan: never
        # coerce truthy strings, positive detections, or contradictory boxes.
        if isinstance(data, dict) and data.get("found") is False and "bbox_2d" in data and data["bbox_2d"] is None:
            data = dict(data)
            confidence = data.get("likelihood")
            if confidence is None or (type(confidence) in (int, float) and confidence == 0):
                data["likelihood"] = "low"
            if data.get("description") is None:
                data["description"] = "Target is not visible in this view"
        return data

    def to_detection(self) -> SearchDetection:
        box = self.bbox_2d
        if box is not None:
            if len(box) != 4:
                raise ValueError("bbox_2d must contain four XYXY coordinates")
            xmin, ymin, xmax, ymax = box
            box = [ymin, xmin, ymax, xmax]
        return SearchDetection(
            found=self.found, likelihood=self.likelihood,
            box_2d=box, description=self.description,
        )


class _CropCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    matches: StrictBool


RESPONSE_INSTRUCTIONS = (
    'Respond with ONLY JSON: {"found":true,"likelihood":"high",'
    '"bbox_2d":[x_min,y_min,x_max,y_max],"description":"short description"}. '
    "bbox_2d is in x/y order with integer coordinates normalized to 0..1000 across the FULL image. "
    "x is horizontal, y is vertical. Top-left is (0,0), bottom-right is (1000,1000). "
    "Halfway down the image is y=500. Use a tight box around the object only. "
    "If absent use found=false and bbox_2d=null. If location is uncertain use bbox_2d=null."
)


def _client() -> OpenAI:
    # Bounded on purpose: the rover sits with its wheels stopped for as long
    # as a call is outstanding, and the SDK defaults to a 10 minute timeout
    # plus silent retries.
    return OpenAI(
        base_url=settings.omni_base_url,
        api_key=settings.omni_api_key,
        timeout=settings.omni_vision_timeout_seconds,
        max_retries=settings.omni_vision_max_retries,
    )


def _normalize_image(image_bytes: bytes) -> bytes:
    """Bake in EXIF orientation so box_2d coordinates are unambiguous.

    Phone photos commonly store pixels in their sensor's native (often
    landscape) orientation plus an EXIF rotation tag. If we pass that
    straight through, it's not guaranteed the model applies the same
    rotation before reasoning about pixel positions -- silently shifting
    every box_2d value. Re-encoding after exif_transpose removes the
    ambiguity: the bytes we send are already in the visually-correct
    orientation, with no rotation metadata left to disagree about.
    """
    image = PILImage.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _image_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    normalized = _normalize_image(image_bytes)
    return f"data:{mime};base64,{base64.b64encode(normalized).decode('ascii')}"


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in model response: {text!r}")
    return json.loads(match.group(0))


def _call(content: list[dict], *, target_text: str) -> DetectionResult:
    client = _client()
    start = time.monotonic()
    response = client.chat.completions.create(
        model=settings.omni_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=500,
        temperature=0.1,
    )
    text = response.choices[0].message.content or ""
    data = _extract_json(text)
    detection = _ModelDetection.model_validate(data).to_detection()
    if detection.description.strip().lower() in {"short description", "...", ""}:
        detection.description = target_text
    if detection.box_2d is not None:
        # Verify actual pixels, independently of the plausible description.
        # An incorrect crop must raise (stationary retry), not become a no-match
        # that would authorize the next scanning turn.
        from app.services.bbox import crop_to_box

        scene_url = content[-1]["image_url"]["url"]
        scene_bytes = base64.b64decode(scene_url.split(",", 1)[1])
        with PILImage.open(io.BytesIO(scene_bytes)) as scene:
            crop = crop_to_box(scene, detection.box_2d)
            buf = io.BytesIO()
            crop.save(buf, format="JPEG")
        remaining = settings.omni_vision_timeout_seconds - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError("Localization verification exceeded the vision time budget")
        verification = client.chat.completions.create(
            model=settings.omni_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    f'The user is looking for: {json.dumps(target_text)}. '
                    "Inspect ONLY the attached crop. Does it visibly contain the requested physical "
                    "object, with matching category and visible attributes? Check the object itself; "
                    "spatial relations to landmarks outside this crop cannot be checked here. "
                    "A blank wall, floor, background, or unrelated object is NOT a match. "
                    "If uncertain answer false. Return ONLY JSON: {\"matches\": true} or {\"matches\": false}."
                )},
                {"type": "image_url", "image_url": {"url": _image_to_data_url(buf.getvalue())}},
            ]}],
            max_tokens=80, temperature=0.1, timeout=remaining,
        )
        check = _CropCheck.model_validate(_extract_json(verification.choices[0].message.content or ""))
        if not check.matches:
            raise ValueError("Localization failed: the proposed crop does not visibly contain the target")
    latency_seconds = time.monotonic() - start
    logger.info("OMNI detection and localization check took %.2fs (model=%s)", latency_seconds, settings.omni_model)
    return DetectionResult(detection=detection, latency_seconds=latency_seconds)


# --- optional hints from the rover controller ---------------------------------
# Both modes accept them as keyword arguments. The response schema does not
# change: still the single strongest candidate, never a list of alternatives.

MAX_REJECTED_HINTS = 3
_MAX_HINT_DESCRIPTION_CHARS = 200

CONFIRMED_LABEL = "USER-CONFIRMED TARGET (earlier view):"
CURRENT_FRAME_LABEL = "CURRENT CAMERA FRAME:"


def _rejected_label(index: int) -> str:
    return f"REJECTED OBJECT {index}:"


def _hint_description(text: str) -> str:
    """One tidy line. These descriptions are model output being fed back into
    a prompt, so keep them short and unable to break out of their quotes."""
    cleaned = " ".join((text or "").split()).replace('"', "'")
    if len(cleaned) > _MAX_HINT_DESCRIPTION_CHARS:
        cleaned = cleaned[: _MAX_HINT_DESCRIPTION_CHARS - 3].rstrip() + "..."
    return cleaned or "(no description)"


def _confirmed_instructions() -> str:
    return (
        f'The image labelled "{CONFIRMED_LABEL}" shows the exact object the user has already '
        "confirmed is the one they want, seen from an earlier camera position. Find that SAME "
        "physical object in the CURRENT CAMERA FRAME -- not merely another object of the same "
        "category. Its apparent size, angle and position may differ because the camera has moved. "
        "If several similar objects are visible and you are unsure which one is the confirmed "
        "object, answer found=false or use a lower likelihood.\n\n"
    )


def _rejected_instructions(rejected: "list[tuple[RejectedHint, str | None]]") -> str:
    lines = ["The user has already REJECTED the following object(s). They are NOT the target:"]
    for idx, (hint, image_url) in enumerate(rejected, start=1):
        shown = " (image below)" if image_url is not None else ""
        lines.append(f'{_rejected_label(idx)} "{_hint_description(hint.description)}"{shown}')
    lines.append(
        "Never report a rejected object again. If the only object visible in the current frame "
        "that matches the description is one of the rejected objects, respond with found=false."
    )
    return "\n".join(lines) + "\n\n"


def _rejected_with_images(rejected: "list[RejectedHint] | None") -> "list[tuple[RejectedHint, str | None]]":
    """The most recent MAX_REJECTED_HINTS hints (most recent last), each with
    its crop as a data URL when it has a usable one. A crop that cannot be
    decoded only costs the picture -- the description still goes in the prompt."""
    prepared: list[tuple[RejectedHint, str | None]] = []
    for hint in list(rejected or [])[-MAX_REJECTED_HINTS:]:
        image_url: str | None = None
        if hint.crop_jpeg:
            try:
                image_url = _image_to_data_url(hint.crop_jpeg)
            except Exception:  # noqa: BLE001
                logger.warning("rejected-object crop could not be decoded; sending its description only")
        prepared.append((hint, image_url))
    return prepared


def build_content(
    target_text: str,
    scene_image: bytes,
    reference_images: list[bytes] | None = None,
    *,
    rejected: "list[RejectedHint] | None" = None,
    confirmed_crop: bytes | None = None,
) -> list[dict]:
    """The chat ``content`` for one detection call. Pure -- no network.

    ``reference_images=None`` builds Mode A (text only); a list builds Mode B
    (personalized). With ``rejected`` and ``confirmed_crop`` both empty the
    result is exactly what detect_text_only / detect_personalized have always
    sent. Otherwise the images go in this order: reference images, the
    user-confirmed crop, rejected objects, and the current camera frame last.
    """
    rejected_hints = _rejected_with_images(rejected)
    # A broken confirmed crop raises here instead of being dropped: it is the
    # identity anchor for an approach, and without it the rover could close in
    # on any object of the right category.
    confirmed_url = _image_to_data_url(confirmed_crop) if confirmed_crop else None

    extra = ""
    if confirmed_url is not None:
        extra += _confirmed_instructions()
    if rejected_hints:
        extra += _rejected_instructions(rejected_hints)

    hint_items: list[dict] = []
    if confirmed_url is not None:
        hint_items.append({"type": "text", "text": CONFIRMED_LABEL})
        hint_items.append({"type": "image_url", "image_url": {"url": confirmed_url}})
    for idx, (_, image_url) in enumerate(rejected_hints, start=1):
        if image_url is not None:
            hint_items.append({"type": "text", "text": _rejected_label(idx)})
            hint_items.append({"type": "image_url", "image_url": {"url": image_url}})

    if reference_images is None:
        prompt = (
            f"Identify the strongest physical match for {target_text} in the image. "
            + extra
            + RESPONSE_INSTRUCTIONS
        )
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend(hint_items)
        if hint_items:
            # Only needed once the frame is no longer the only image.
            content.append({"type": "text", "text": CURRENT_FRAME_LABEL})
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(scene_image)}})
        return content

    if reference_images or confirmed_url is None:
        intro = (
            f'You are looking for a specific object described as: "{target_text}".\n'
            f"Below are {len(reference_images)} REFERENCE IMAGE(S) showing exactly the specific item "
            "to find -- note its precise color, shape, logos, attached objects, and proportions. "
            "Then a CURRENT CAMERA FRAME follows. Find the SAME specific object in the current frame, "
            "not just any object of the same category. Category membership alone is NOT sufficient -- "
            "a different item of the same general type (e.g. a different keyring) is not a match unless "
            "its visual details genuinely match the reference images.\n\n"
        )
    else:
        # Approach re-identification of an item that has no reference photos:
        # the user-confirmed crop is the only picture of the target.
        intro = (
            f'You are looking for a specific object described as: "{target_text}".\n'
            "Category membership alone is NOT sufficient -- a different item of the same general "
            "type is not a match.\n\n"
        )
    content = [{"type": "text", "text": intro + extra + RESPONSE_INSTRUCTIONS}]
    for idx, ref in enumerate(reference_images, start=1):
        content.append({"type": "text", "text": f"REFERENCE IMAGE {idx}:"})
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(ref)}})
    content.extend(hint_items)
    content.append({"type": "text", "text": CURRENT_FRAME_LABEL})
    content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(scene_image)}})
    return content


def detect_text_only(
    target_text: str,
    scene_image: bytes,
    *,
    rejected: "list[RejectedHint] | None" = None,
    confirmed_crop: bytes | None = None,
) -> DetectionResult:
    """Mode A: target text + one static current-scene image, no reference photos."""
    return _call(
        build_content(target_text, scene_image, rejected=rejected, confirmed_crop=confirmed_crop),
        target_text=target_text,
    )


def detect_personalized(
    target_text: str,
    reference_images: list[bytes],
    scene_image: bytes,
    *,
    rejected: "list[RejectedHint] | None" = None,
    confirmed_crop: bytes | None = None,
) -> DetectionResult:
    """Mode B: target text + 1-3 reference photos of the specific item + one current-scene image."""
    return _call(
        build_content(
            target_text, scene_image, list(reference_images), rejected=rejected, confirmed_crop=confirmed_crop
        ),
        target_text=target_text,
    )
