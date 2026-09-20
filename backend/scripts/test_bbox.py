import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageOps

from app.services.bbox import box_2d_to_pixels, crop_to_box


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug tool for the bbox crop utility (Spec 10).")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--box",
        required=True,
        nargs=4,
        type=int,
        metavar=("YMIN", "XMIN", "YMAX", "XMAX"),
        help="box_2d on the 0-1000 scale",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    image = Image.open(args.image)
    image = ImageOps.exif_transpose(image).convert("RGB")

    pixel_box = box_2d_to_pixels(args.box, image.width, image.height)
    print(f"image size (EXIF-corrected): {image.width}x{image.height}")
    print(f"box_2d {args.box} -> pixels: {pixel_box}")

    crop = crop_to_box(image, args.box)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(args.out)
    print(f"crop size: {crop.width}x{crop.height}")
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
