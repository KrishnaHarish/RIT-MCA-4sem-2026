"""
Create a minimal demo model with random weights for UI testing.

Usage:
    python src/create_demo_model.py
    python src/create_demo_model.py --arch resnet50 --output_dir models

This lets you test the Streamlit app (app/app.py) without running a full training job.
The model predicts randomly (untrained weights), but demonstrates the complete inference UI.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import models


DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_maize___Common_rust_",
    "Corn_maize___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]


def get_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = getattr(models, arch)(weights=None)
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            raise ValueError(f"Don't know how to replace classifier head for arch='{arch}'.")
    return model


def main():
    parser = argparse.ArgumentParser(description="Create a demo model with random weights for UI testing")
    parser.add_argument("--arch", type=str, default="resnet18", help="Model backbone (resnet18 or resnet50)")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the model artifacts")
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Path to an existing classes.json (uses built-in default classes if omitted)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.classes:
        classes_path = Path(args.classes)
        if not classes_path.exists():
            raise FileNotFoundError(f"classes file not found: {classes_path}")
        classes = json.loads(classes_path.read_text(encoding="utf-8"))
    else:
        classes = DEFAULT_CLASSES

    num_classes = len(classes)
    model = get_model(args.arch, num_classes=num_classes)
    model.eval()

    classes_out = output_dir / "classes.json"
    model_out = output_dir / "model.pt"

    classes_out.write_text(json.dumps(classes, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), model_out)

    print(f"Demo model saved to: {model_out}")
    print(f"Classes saved to:    {classes_out}")
    print(f"Architecture:        {args.arch}")
    print(f"Number of classes:   {num_classes}")
    print()
    print("NOTE: This model has random weights (not trained). Predictions will be random.")
    print("Run the Streamlit app with:")
    print("    streamlit run app/app.py")


if __name__ == "__main__":
    main()
