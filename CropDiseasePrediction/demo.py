"""
Quick Demo — Crop Disease Prediction
=====================================
Trains a tiny ResNet-18 model for 1 epoch on the bundled `data_smoke/`
dataset (2 classes, ~1 k images) and immediately runs inference on a
random validation image.

Usage (from the CropDiseasePrediction directory):
    python demo.py

No arguments required. The demo writes a temporary model artifact to
`/tmp/demo_model/` and does NOT modify any committed files.
"""

import json
import random
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 1
LR = 1e-3
TOP_K = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model(num_classes: int) -> nn.Module:
    """Return a ResNet-18 with ImageNet weights and a replaced FC head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze backbone — only train the final classification head.
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total)


def predict(model, image_path: Path, classes: list, device, top_k: int = TOP_K):
    tfm = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)
    k = min(top_k, len(classes))
    values, indices = torch.topk(probs, k=k)
    return [(classes[class_idx], prob) for class_idx, prob in zip(indices.tolist(), values.tolist())]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data_smoke"

    if not data_dir.exists():
        print(
            f"[ERROR] Smoke dataset not found at: {data_dir}\n"
            "Please ensure the repository contains the 'data_smoke/' folder.",
            file=sys.stderr,
        )
        sys.exit(1)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    # ---- Data loading -------------------------------------------------------
    train_tfms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tfms)
    classes = train_ds.classes
    num_classes = len(classes)

    print("=" * 60)
    print("  Crop Disease Prediction - Quick Demo")
    print("=" * 60)
    print(f"  Classes ({num_classes}): {', '.join(classes)}")
    print(f"  Train images : {len(train_ds)}")
    print(f"  Val images   : {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Model / training ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device       : {device}")
    print()

    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    print(f"[1/3] Training for {EPOCHS} epoch(s) ...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, val_loader, device)
        print(f"      Epoch {epoch}/{EPOCHS} - loss={loss:.4f}  val_acc={acc:.4f}")

    # ---- Save artifacts to a temp directory --------------------------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "model.pt"
        classes_path = tmp_path / "classes.json"

        torch.save(model.state_dict(), model_path)
        classes_path.write_text(json.dumps(classes, indent=2), encoding="utf-8")

        print(f"\n[2/3] Model saved to temporary directory.")

        # ---- Pick a random val image ----------------------------------------
        all_val_images = [
            p for p in val_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not all_val_images:
            print("[WARNING] No validation images found; skipping inference demo.")
            return

        sample_image = random.choice(all_val_images)
        true_label = sample_image.parent.name

        print(f"\n[3/3] Running inference on a random validation image ...")
        print(f"      File  : {sample_image.name}")
        print(f"      Label : {true_label}")

        model.eval()
        results = predict(model, sample_image, classes, device)

        print()
        print("  Top predictions:")
        for rank, (cls, prob) in enumerate(results, start=1):
            marker = " [OK]" if cls == true_label else ""
            print(f"    {rank}. {cls:<40s} {prob:.4f}{marker}")

    print()
    print("=" * 60)
    print("  Demo complete!  The trained model was stored in a temporary")
    print("  directory and has been cleaned up automatically.")
    print("  To train a full model, see the README for instructions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
