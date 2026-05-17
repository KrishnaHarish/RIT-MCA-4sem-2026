import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from torchvision import datasets, models, transforms


def main():
    data_dir = Path("data_tomato_small")
    out_dir = Path("models_tomato_small")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_tfms)
    val_ds = datasets.ImageFolder(str(data_dir / "val"), transform=val_tfms)
    classes = train_ds.classes

    # class-balanced sampler to reduce bias towards dominant classes
    class_counts = Counter(train_ds.targets)
    sample_weights = [1.0 / class_counts[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pretrained ResNet18 with transfer learning; fine-tune last block + head
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    for name, p in model.named_parameters():
        p.requires_grad = False
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    params_to_opt = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_opt, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epochs = 10
    for epoch in range(1, epochs + 1):
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

        avg_loss = running_loss / len(train_ds)
        print(f"Epoch {epoch}/{epochs} train_loss={avg_loss:.4f}")
        scheduler.step()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)

    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    metrics = {
        "dataset": "data_tomato_small",
        "num_classes": len(classes),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "confusion_matrix": cm.tolist(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    print("Confusion matrix:")
    print(cm)
    print(f"Saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
