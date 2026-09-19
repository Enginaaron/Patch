import io
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import settings


class InvalidImageError(Exception):
    pass


async def save_reference_image(item_id: str, upload: UploadFile) -> str:
    contents = await upload.read()
    try:
        Image.open(io.BytesIO(contents)).verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"{upload.filename} is not a valid image") from exc

    ext = Path(upload.filename or "").suffix.lower() or ".jpg"
    item_dir = Path(settings.media_root) / "items" / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4()}{ext}"
    (item_dir / filename).write_bytes(contents)
    return f"items/{item_id}/{filename}"
