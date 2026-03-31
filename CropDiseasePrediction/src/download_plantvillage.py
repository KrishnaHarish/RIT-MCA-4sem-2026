import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def label_to_name(label, label_features) -> str:
    # Hugging Face usually stores labels either as strings or as ClassLabel (int + names).
    try:
        # ClassLabel case
        if hasattr(label_features, "names"):
            if isinstance(label, int):
                return label_features.names[label]
        # String case
        if isinstance(label, str):
            return label
    except Exception:
        pass

    return str(label)


def main():
    parser = argparse.ArgumentParser(description="Download PlantVillage from Hugging Face and export to ImageFolder structure.")
    parser.add_argument("--out_dir", type=str, default="data", help="Output folder to create train/ and val/ inside.")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Dataset config name on Hugging Face Hub (for mohanthy/PlantVillage this is typically 'default').",
    )
    parser.add_argument("--max_train_images", type=int, default=0, help="0 means download all train images.")
    parser.add_argument("--max_val_images", type=int, default=0, help="0 means download all test images as val.")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    train_root = out_root / "train"
    val_root = out_root / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    print("Downloading PlantVillage from Hugging Face...")
    ds = load_dataset("mohanty/PlantVillage", args.config)

    label_features = ds["train"].features["label"]

    # Export train -> train, test -> val
    split_map = {"train": ("train", args.max_train_images), "test": ("val", args.max_val_images)}

    for split_name, (dest_split, max_images) in split_map.items():
        cur_ds = ds[split_name]
        total = len(cur_ds)
        limit = total if max_images <= 0 else min(total, max_images)
        print(f"Exporting {split_name} -> {dest_split} ({limit}/{total})")

        for i in tqdm(range(limit), desc=f"{split_name}->{dest_split}"):
            ex = cur_ds[i]
            img = ex["image"]  # PIL.Image
            label = ex["label"]
            class_name = label_to_name(label, label_features)

            dest_class_dir = (train_root if dest_split == "train" else val_root) / class_name
            dest_class_dir.mkdir(parents=True, exist_ok=True)

            # Use a deterministic filename for reproducibility.
            out_path = dest_class_dir / f"{split_name}_{i:06d}.jpg"
            img.save(out_path, format="JPEG", quality=95)

    print("Done.")
    print(f"Train folder: {train_root}")
    print(f"Val folder:   {val_root}")


if __name__ == "__main__":
    main()

