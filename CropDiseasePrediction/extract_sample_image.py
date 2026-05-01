#!/usr/bin/env python3
"""Decode embedded base64 payload into `models/sample.jpg`.

Run from the repo root:
  python CropDiseasePrediction/extract_sample_image.py
This writes `CropDiseasePrediction/models/sample.jpg`.
"""
import base64
from pathlib import Path

PAYLOAD_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAAQABADASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAGf/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPwA//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwA//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwA//9k="
)


def main():
    out = Path(__file__).resolve().parents[0] / "models" / "sample.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = base64.b64decode(PAYLOAD_B64)
    out.write_bytes(data)
    print(f"Wrote sample image to: {out}")


if __name__ == "__main__":
    main()
