import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def main():
    classes = json.loads(Path("models/classes.json").read_text(encoding="utf-8"))
    val_dir = Path("data_smoke/val") if Path("data_smoke/val").exists() else Path("data/val")

    tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = datasets.ImageFolder(str(val_dir), transform=tfms)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state = torch.load("models/model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    n = len(classes)
    cm = [[0 for _ in range(n)] for _ in range(n)]
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader, start=1):
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            for yt, yp in zip(y.tolist(), preds.tolist()):
                cm[yt][yp] += 1
                correct += int(yt == yp)
                total += 1
            print(f"Processed batch {idx}/{len(loader)}")

    accuracy = correct / total if total else 0.0

    # Macro F1 from confusion matrix (manual; avoids extra dependencies).
    f1_scores = []
    for i in range(n):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_scores.append(f1)
    f1_macro = sum(f1_scores) / n if n else 0.0

    result = {
        "classes": ds.classes,
        "val_dir": str(val_dir),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "confusion_matrix": cm,
    }
    Path("models/metrics_current.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
