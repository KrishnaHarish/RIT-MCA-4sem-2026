#!/usr/bin/env python3
"""Generate a small sample image at `models/sample.jpg` for demos.

Usage:
  python generate_sample_image.py
This creates `CropDiseasePrediction/models/sample.jpg` (64x64 RGB).
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    out = Path(__file__).resolve().parents[0] / "models" / "sample.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (64, 64), color=(200, 230, 200))
    d = ImageDraw.Draw(img)
    # Draw a simple leaf-like shape
    d.ellipse((8, 8, 56, 56), fill=(34, 139, 34))
    d.line((32, 8, 32, 56), fill=(0, 100, 0), width=2)

    img.save(out, quality=85)
    print(f"Wrote sample image to: {out}")


if __name__ == "__main__":
    main()
