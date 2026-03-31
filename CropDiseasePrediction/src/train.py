import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower().strip()

    # Transfer learning: load ImageNet weights for common backbones, then replace classifier head.
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        # Fallback: create model without pretrained weights.
        model = getattr(models, arch)(weights=None)
        # Heuristic for torchvision models: replace the final fully-connected layer if present.
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Don't know how to replace classifier head for arch='{arch}'.")

    # Freeze backbone for faster training on small datasets; only train the head.
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze classifier head
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    return model


def main():
    parser = argparse.ArgumentParser(description="Train crop disease image classifier")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Either: (1) folder with train/ and val/ subfolders, or (2) PlantVillage-style root where class folders are directly inside data_dir.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Only used when data_dir has no train/val split.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed (when auto-splitting PlantVillage).")
    parser.add_argument("--output_dir", type=str, default="models", help="Where to save model artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    if train_dir.exists() and val_dir.exists():
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
        val_ds = datasets.ImageFolder(str(val_dir), transform=val_tfms)

        # Ensure class set matches between train/val.
        if set(train_ds.classes) != set(val_ds.classes):
            raise ValueError("Class names mismatch between train and val folders.")

        classes = train_ds.classes
        num_classes = len(classes)
    else:
        # PlantVillage-style layout: data_dir/class_name/*.jpg (no pre-made train/val).
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

        if not (data_dir / "class_1").exists():
            # We don't know class folder names up-front; just ensure ImageFolder sees something.
            pass

        full_train_base = datasets.ImageFolder(str(data_dir), transform=train_tfms)
        full_val_base = datasets.ImageFolder(str(data_dir), transform=val_tfms)
        classes = full_train_base.classes
        num_classes = len(classes)
        total = len(full_train_base)

        if total == 0:
            raise ValueError(f"No images found under: {data_dir}")

        if not (0.0 < args.val_ratio < 1.0):
            raise ValueError("--val_ratio must be between 0 and 1 (exclusive).")

        g = torch.Generator()
        g.manual_seed(args.seed)
        indices = torch.randperm(total, generator=g).tolist()

        val_size = int(total * args.val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_ds = Subset(full_train_base, train_indices)
        val_ds = Subset(full_val_base, val_indices)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.arch, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_acc = 0.0
    best_path = output_dir / "model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(1, total)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"Training complete. Best val_acc={best_acc:.4f}")
    # Ensure we always produce a model artifact, even if early epochs never exceed best_acc.
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()

