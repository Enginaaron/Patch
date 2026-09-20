from dataclasses import dataclass

from PIL import Image, ImageOps


@dataclass
class PixelBox:
    left: int
    top: int
    right: int
    bottom: int


def box_2d_to_pixels(box_2d: list[int], image_width: int, image_height: int) -> PixelBox:
    """Convert [ymin, xmin, ymax, xmax] on the 0-1000 scale (Spec 7) to clamped pixel coords."""
    ymin, xmin, ymax, xmax = box_2d

    def scale(value: int, dimension: int) -> int:
        pixel = round(value / 1000 * dimension)
        return max(0, min(dimension, pixel))

    left = scale(xmin, image_width)
    top = scale(ymin, image_height)
    right = scale(xmax, image_width)
    bottom = scale(ymax, image_height)

    # Clamping each edge independently can collapse a degenerate/edge-hugging
    # box to zero width or height -- keep at least 1px so callers never get an
    # empty crop.
    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)

    return PixelBox(left=left, top=top, right=right, bottom=bottom)


def crop_to_box(image: Image.Image, box_2d: list[int]) -> Image.Image:
    """Crop `image` to `box_2d`.

    box_2d coordinates are always relative to the EXIF-corrected (visually
    upright) orientation -- that's what the vision model reasoned against
    (see omni_vision._normalize_image). PIL's Image.open does not apply EXIF
    rotation on its own, so an un-rotated phone photo would silently crop the
    wrong region without this -- the same bug Spec 7 fixed for the model
    call, reintroduced here if skipped.
    """
    image = ImageOps.exif_transpose(image)
    box = box_2d_to_pixels(box_2d, image.width, image.height)
    return image.crop((box.left, box.top, box.right, box.bottom))
