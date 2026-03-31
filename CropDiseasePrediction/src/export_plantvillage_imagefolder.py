import argparse
import os
import shutil
from pathlib import Path


def export_split(lines, extracted_color_root: Path, out_split_root: Path, max_images: int, split_name: str):
    out_split_root.mkdir(parents=True, exist_ok=True)
    written = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Format: <full_path_to_color/Class/file> \t <int_label>
        parts = line.split("\t", 1)
        src_full = parts[0]

        # src_full looks like:
        # /home/h/Downloads/plantvillage_deeplearning_paper_dataset/color/<ClassName>/<filename>
        if "/color/" not in src_full:
            # Unexpected format; skip
            continue

        rel_after_color = src_full.split("/color/", 1)[1]
        class_name = rel_after_color.split("/", 1)[0]
        filename = rel_after_color.split("/", 1)[1]

        src_path = extracted_color_root / class_name / filename
        if not src_path.exists():
            # Some filenames may have odd whitespace differences; if missing, skip.
            continue

        dest_class_dir = out_split_root / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_class_dir / filename

        if not dest_path.exists():
            # Default to copy; can be overridden via --link_mode by passing a global function.
            # We keep the implementation simple: this function is called only after we set mode.
            mode = getattr(export_split, "_link_mode", "copy")
            if mode == "hardlink":
                try:
                    os.link(src_path, dest_path)
                except Exception:
                    shutil.copy2(src_path, dest_path)
            elif mode == "copy":
                shutil.copy2(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)

        written += 1
        if max_images > 0 and written >= max_images:
            break

    print(f"{split_name}: exported {written} images -> {out_split_root}")


def main():
    parser = argparse.ArgumentParser(description="Export PlantVillage into ImageFolder train/val layout.")
    parser.add_argument(
        "--extracted_root",
        type=str,
        default=str(Path("data") / "plantvillage_extracted" / "plantvillage_deeplearning_paper_dataset"),
        help="Path to the extracted dataset root (contains color/ and 80-20/ folders).",
    )
    parser.add_argument("--split_name", type=str, default="80-20", help="Which split folder to read: 10-90, 20-80, 80-20, etc.")
    parser.add_argument("--out_dir", type=str, default=str(Path("data")), help="Output root (creates train/ and val/).")
    parser.add_argument("--max_train_images", type=int, default=0, help="0 = all train images")
    parser.add_argument("--max_val_images", type=int, default=0, help="0 = all val images (from valid.txt)")
    parser.add_argument(
        "--link_mode",
        type=str,
        default="copy",
        choices=["copy", "hardlink"],
        help="Use hardlinks to avoid duplicating files (recommended on same drive).",
    )
    args = parser.parse_args()

    extracted_root = Path(args.extracted_root)
    extracted_color_root = extracted_root / "color"
    split_root = extracted_root / args.split_name

    train_txt = split_root / "train.txt"
    valid_txt = split_root / "valid.txt"

    if not extracted_color_root.exists():
        raise FileNotFoundError(f"Missing color folder: {extracted_color_root}")
    if not train_txt.exists():
        raise FileNotFoundError(f"Missing train.txt: {train_txt}")
    if not valid_txt.exists():
        raise FileNotFoundError(f"Missing valid.txt: {valid_txt}")

    # Export to ImageFolder layout
    out_train_root = Path(args.out_dir) / "train"
    out_val_root = Path(args.out_dir) / "val"

    export_split._link_mode = args.link_mode  # type: ignore[attr-defined]

    # Read lines once (these files can be large, but manageable for our use).
    train_lines = train_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    valid_lines = valid_txt.read_text(encoding="utf-8", errors="ignore").splitlines()

    export_split(
        lines=train_lines,
        extracted_color_root=extracted_color_root,
        out_split_root=out_train_root,
        max_images=args.max_train_images,
        split_name="train",
    )
    export_split(
        lines=valid_lines,
        extracted_color_root=extracted_color_root,
        out_split_root=out_val_root,
        max_images=args.max_val_images,
        split_name="val",
    )


if __name__ == "__main__":
    main()

