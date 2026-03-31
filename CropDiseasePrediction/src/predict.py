import argparse
import json
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image
from torch import nn


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model = getattr(models, arch)(weights=None)
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Don't know how to replace classifier head for arch='{arch}'.")
    return model


def main():
    parser = argparse.ArgumentParser(description="Predict crop disease from an image")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--classes_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    classes_path = Path(args.classes_path)
    image_path = Path(args.image_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing classes file: {classes_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file: {image_path}")

    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.arch, num_classes=num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    topk = min(args.top_k, num_classes)
    values, indices = torch.topk(probs, k=topk)

    print("Top predictions:")
    for v, idx in zip(values.tolist(), indices.tolist()):
        print(f"- {classes[idx]}: {v:.4f}")


if __name__ == "__main__":
    main()

