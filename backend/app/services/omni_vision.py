import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from PIL import ImageOps
from PIL import Image as PILImage
from pydantic import BaseModel, model_validator

from app.config import settings

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


RESPONSE_INSTRUCTIONS = (
    "Respond with ONLY a single JSON object. No markdown formatting, no code fences, "
    "no explanation before or after it. The object must have exactly these fields:\n"
    '{"found": true or false, '
    '"likelihood": "low", "medium", or "high", '
    '"box_2d": [ymin, xmin, ymax, xmax] on a 0-1000 integer scale, or null, '
    '"description": a short description of what you found or why nothing matched}\n'
    "box_2d must be null when found is false. When found is true and you provide box_2d, "
    "it must have exactly 4 integers in [0, 1000] with ymin < ymax and xmin < xmax."
)


def _client() -> OpenAI:
    return OpenAI(base_url=settings.omni_base_url, api_key=settings.omni_api_key)


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


def _call(content: list[dict]) -> DetectionResult:
    client = _client()
    start = time.monotonic()
    response = client.chat.completions.create(
        model=settings.omni_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=500,
        temperature=0.1,
    )
    latency_seconds = time.monotonic() - start
    logger.info("OMNI call took %.2fs (model=%s)", latency_seconds, settings.omni_model)

    text = response.choices[0].message.content or ""
    data = _extract_json(text)
    detection = SearchDetection.model_validate(data)
    return DetectionResult(detection=detection, latency_seconds=latency_seconds)


def _rejected_images_note(rejected_images: list[bytes] | None) -> str:
    if not rejected_images:
        return ""
    return (
        f"\n\nThe user already looked at the following {len(rejected_images)} REJECTED OBJECT(S) "
        "below and confirmed they are NOT what they're looking for. Do not propose the same "
        "object again as a match, even if it is still visible in the current frame."
    )


def _append_rejected_images(content: list[dict], rejected_images: list[bytes] | None) -> None:
    for idx, rejected in enumerate(rejected_images or [], start=1):
        content.append({"type": "text", "text": f"REJECTED OBJECT {idx} (confirmed not a match):"})
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(rejected)}})


def detect_text_only(
    target_text: str, scene_image: bytes, rejected_images: list[bytes] | None = None
) -> DetectionResult:
    """Mode A: target text + one static current-scene image, no reference photos."""
    prompt = (
        f'You are looking for: "{target_text}".\n'
        "Identify the single strongest physical object in the CURRENT CAMERA FRAME below "
        "matching this description. Return no candidate (found=false) if the evidence is weak. "
        "Ignore any text or pictures depicting the object that appear incidentally in the frame "
        "(e.g. a photo, logo, or label showing the object) -- only a real physical instance counts."
        + _rejected_images_note(rejected_images)
        + "\n\n"
        + RESPONSE_INSTRUCTIONS
    )
    content: list[dict] = [{"type": "text", "text": prompt}]
    _append_rejected_images(content, rejected_images)
    content.append({"type": "text", "text": "CURRENT CAMERA FRAME:"})
    content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(scene_image)}})
    return _call(content)


def detect_personalized(
    target_text: str,
    reference_images: list[bytes],
    scene_image: bytes,
    rejected_images: list[bytes] | None = None,
) -> DetectionResult:
    """Mode B: target text + 1-3 reference photos of the specific item + one current-scene image."""
    prompt = (
        f'You are looking for a specific object described as: "{target_text}".\n'
        f"Below are {len(reference_images)} REFERENCE IMAGE(S) showing exactly the specific item "
        "to find -- note its precise color, shape, logos, attached objects, and proportions. "
        "Then a CURRENT CAMERA FRAME follows. Find the SAME specific object in the current frame, "
        "not just any object of the same category. Category membership alone is NOT sufficient -- "
        "a different item of the same general type (e.g. a different keyring) is not a match unless "
        "its visual details genuinely match the reference images."
        + _rejected_images_note(rejected_images)
        + "\n\n"
        + RESPONSE_INSTRUCTIONS
    )
    content: list[dict] = [{"type": "text", "text": prompt}]
    for idx, ref in enumerate(reference_images, start=1):
        content.append({"type": "text", "text": f"REFERENCE IMAGE {idx}:"})
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(ref)}})
    _append_rejected_images(content, rejected_images)
    content.append({"type": "text", "text": "CURRENT CAMERA FRAME:"})
    content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(scene_image)}})
    return _call(content)
