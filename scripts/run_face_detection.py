#!/usr/bin/env python3
"""
Helper script to run face emotion detection on an image.
Usage:
  python scripts/run_face_detection.py /path/to/image.jpg
"""
import argparse
import json
from pathlib import Path

from models.face_emotion import detect_face_emotion


def main():
    parser = argparse.ArgumentParser(description="Run face emotion detection on an image using DeepFace.")
    parser.add_argument("image_path", help="Path to the image file (jpg/png/jpeg)")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(json.dumps({"error": f"File not found: {img_path}"}))
        raise SystemExit(1)

    with open(img_path, "rb") as f:
        result = detect_face_emotion(f)

    print(json.dumps(result, indent=2))
    if "error" in result:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
