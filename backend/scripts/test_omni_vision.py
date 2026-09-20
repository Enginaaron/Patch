import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.omni_vision import detect_personalized, detect_text_only


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug tool for the OMNI vision service (Spec 7).")
    sub = parser.add_subparsers(dest="mode", required=True)

    mode_a = sub.add_parser("mode-a", help="Text-only detection")
    mode_a.add_argument("--target", required=True)
    mode_a.add_argument("--scene", required=True, type=Path)

    mode_b = sub.add_parser("mode-b", help="Personalized detection")
    mode_b.add_argument("--target", required=True)
    mode_b.add_argument("--scene", required=True, type=Path)
    mode_b.add_argument("--reference", required=True, type=Path, action="append")

    args = parser.parse_args()

    if args.mode == "mode-a":
        result = detect_text_only(args.target, args.scene.read_bytes())
    else:
        if not 1 <= len(args.reference) <= 3:
            parser.error("--reference must be given 1-3 times")
        result = detect_personalized(
            args.target,
            [p.read_bytes() for p in args.reference],
            args.scene.read_bytes(),
        )

    print(result.detection.model_dump_json(indent=2))
    print(f"latency: {result.latency_seconds:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
