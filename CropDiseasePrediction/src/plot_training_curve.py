"""Plot training/validation accuracy curve (quick smoke run).

This script runs a short training session on the bundled `data_smoke/`
dataset and saves a PNG with validation accuracy per epoch to
`models/plots/train_val_accuracy.png`.

Usage (from CropDiseasePrediction/):
    python src/plot_training_curve.py --epochs 10
"""
import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
BATCH_SIZE = 16
LR = 1e-3


def get_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_smoke", help="dataset folder (ImageFolder train/val)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", default="models/plots/train_val_accuracy.png")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / args.data_dir
    if not data_dir.exists():
        raise SystemExit(f"data folder not found: {data_dir}")

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    val_accs = []
    train_losses = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={loss:.4f} | val_acc={acc:.4f}")
        train_losses.append(loss)
        val_accs.append(acc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, args.epochs + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, val_accs, marker="o", label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy per Epoch")
    plt.grid(alpha=0.3)
    plt.xticks(epochs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
