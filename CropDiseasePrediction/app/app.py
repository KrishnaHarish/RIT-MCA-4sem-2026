from pathlib import Path
import json

import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from torch import nn


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


@st.cache_resource
def load_artifacts(project_root: Path, arch: str):
    model_path = project_root / "models" / "model.pt"
    classes_path = project_root / "models" / "classes.json"

    if not model_path.exists() or not classes_path.exists():
        return None, None

    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(arch, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, classes


def main():
    st.set_page_config(page_title="Crop Disease Prediction", layout="centered")
    st.title("Crop Disease Prediction")

    project_root = Path(__file__).resolve().parents[1]
    arch = st.sidebar.selectbox("Model backbone", ["resnet18", "resnet50"], index=0)

    model, classes = load_artifacts(project_root, arch=arch)
    if model is None:
        st.warning("Model not found. Train the model first (see README in this folder).")
        return

    file = st.file_uploader("Upload a leaf/plant image", type=["jpg", "jpeg", "png"])
    if file is None:
        return

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Input image", use_column_width=True)

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_prob, top_idx = torch.max(probs, dim=0)
    st.success(f"Predicted: {classes[top_idx]} (confidence: {top_prob:.3f})")


if __name__ == "__main__":
    main()

