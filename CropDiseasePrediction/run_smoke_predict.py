#!/usr/bin/env python3
"""Lightweight smoke prediction script for CropDiseasePrediction.

This script does NOT require PyTorch. It loads `models/classes.json` (if
present) and computes a deterministic pseudo-prediction from the image bytes
so you can verify the inference pipeline without heavy dependencies.
"""
import argparse
import json
import hashlib
from pathlib import Path
from PIL import Image


def load_classes(path: Path):
    if not path.exists():
        return ["class_0"]
    return json.loads(path.read_text(encoding="utf-8"))


def pseudo_predict(image_path: Path, classes: list):
    # Read bytes and hash to pick a class deterministically
    data = image_path.read_bytes()
    h = int(hashlib.sha256(data).hexdigest(), 16)
    idx = h % len(classes)
    # Map part of the hash to a pseudo-confidence in [0.5, 0.999]
    conf = 0.5 + (h % 500) / 1000.0
    return classes[idx], conf


def main():
    parser = argparse.ArgumentParser(description="Smoke predict without torch")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--classes",
        "-c",
        type=str,
        default="models/classes.json",
        help="Path to classes.json (default: models/classes.json)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    classes_path = Path(args.classes)

    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        raise SystemExit(2)

    try:
        # Validate image can be opened
        Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error: cannot open image: {e}")
        raise SystemExit(3)

    classes = load_classes(classes_path)
    pred, conf = pseudo_predict(image_path, classes)
    print(f"Predicted: {pred} (confidence: {conf:.3f})")


if __name__ == "__main__":
    main()
