"""Create a montage showing one sample image per Tomato class.

The script reads class names from `models/classes.json` (fallback to
`models_tomato/classes.json`) and looks for class folders under a
specified dataset root (default: `data_tomato_small`). It will display
or save a grid image with one sample per class when available.

Usage:
    python src/show_tomato_class_samples.py --data_dir data_smoke --out models/plots/tomato_samples.png
"""
from pathlib import Path
import json
import argparse
import math
import sys

from PIL import Image
import matplotlib.pyplot as plt


def load_classes(repo_root: Path):
    candidates = [repo_root / "models" / "classes.json", repo_root / "models_tomato" / "classes.json"]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def find_sample_for_class(data_root: Path, class_name: str):
    # Check root, then train/ and val/ subfolders
    candidates = [data_root, data_root / "train", data_root / "val"]
    for base in candidates:
        folder = base / class_name
        if not folder.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
            files = list(folder.glob(ext))
            if files:
                return files[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_tomato_small", help="dataset root containing class subfolders")
    parser.add_argument("--out", default="models/plots/tomato_samples.png", help="output image path")
    parser.add_argument("--ncols", type=int, default=5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    classes = load_classes(repo_root)
    if not classes:
        print("No classes.json found under models/ or models_tomato/.", file=sys.stderr)
        sys.exit(1)

    data_root = repo_root / args.data_dir
    if not data_root.exists():
        print(f"Data folder not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    samples = []
    missing = []
    for cls in classes:
        sample = find_sample_for_class(data_root, cls)
        if sample:
            samples.append((cls, sample))
        else:
            missing.append(cls)

    if not samples:
        print("No sample images found for any classes under", data_root)
        sys.exit(1)

    n = len(samples)
    ncols = args.ncols
    nrows = math.ceil(n / ncols)

    plt.figure(figsize=(ncols * 3, nrows * 3))
    for i, (cls, path) in enumerate(samples, start=1):
        img = Image.open(path).convert("RGB")
        plt.subplot(nrows, ncols, i)
        plt.imshow(img)
        plt.title(cls.replace("Tomato___", ""))
        plt.axis("off")

    plt.tight_layout()
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved sample montage to: {out_path}")
    if missing:
        print(f"Warning: {len(missing)} classes missing images under {data_root}." )


if __name__ == "__main__":
    main()
