from pathlib import Path
import json
from io import BytesIO
from urllib.request import urlopen

import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from torch import nn


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_ARCH = "resnet18"
TOP_K = 1

TOMATO_SAMPLE_IMAGES = {
    "Tomato___Bacterial_spot": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Bacterial_spot/00416648-be6e-4bd4-bc8d-82f43f8a7240___GCREC_Bact.Sp%203110.JPG",
    "Tomato___Early_blight": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Early_blight/0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B%208389.JPG",
    "Tomato___Late_blight": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Late_blight/0003faa8-4b27-4c65-bf42-6d9e352ca1a5___RS_Late.B%204946.JPG",
    "Tomato___Leaf_Mold": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Leaf_Mold/00694db7-3327-45e0-b4da-a8bb7ab6a4b7___Crnl_L.Mold%206923.JPG",
    "Tomato___Septoria_leaf_spot": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Septoria_leaf_spot/002533c1-722b-44e5-9d2e-91f7747b2543___Keller.St_CG%201831.JPG",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Spider_mites%20Two-spotted_spider_mite/002835d1-c18e-4471-aa6e-8d8c29585e9b___Com.G_SpM_FL%208584.JPG",
    "Tomato___Target_Spot": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Target_Spot/002213fb-b620-4593-b9ac-6a6cc119b100___Com.G_TgS_FL%208360.JPG",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Tomato_Yellow_Leaf_Curl_Virus/00139ae8-d881-4edb-925f-46584b0bd68c___YLCV_NREC%202944.JPG",
    "Tomato___Tomato_mosaic_virus": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Tomato_mosaic_virus/000ec6ea-9063-4c33-8abe-d58ca8a88878___PSU_CG%202169.JPG",
    "Tomato___healthy": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___healthy/000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL%20Leaf%20259.1.JPG",
}


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


def load_json_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def readable_label(class_name: str) -> str:
    return class_name.replace("Tomato___", "").replace("_", " ")


@st.cache_data(show_spinner=False)
def load_remote_image(url: str) -> bytes:
    with urlopen(url, timeout=15) as response:
        return response.read()


@st.cache_resource
def load_artifacts(project_root: Path):
    model_path = project_root / "models" / "model.pt"
    classes_path = project_root / "models" / "classes.json"

    if not model_path.exists() or not classes_path.exists():
        return None, None, "Model artifact or classes file is missing."

    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(MODEL_ARCH, num_classes=len(classes)).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as exc:
        return None, classes, str(exc)

    model.eval()
    return model, classes, None


def predict(image: Image.Image, model: nn.Module, classes: list[str]):
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    device = next(model.parameters()).device
    x = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    values, indices = torch.topk(probs, k=min(TOP_K, len(classes)))
    return [(classes[idx], float(prob)) for prob, idx in zip(values.tolist(), indices.tolist())]


def show_prediction(results, expected_class: str | None = None):
    predicted_class, confidence = results[0]
    predicted_label = readable_label(predicted_class)
    confidence_text = f"{confidence:.3f}"

    if expected_class is None:
        st.success(f"Predicted: {predicted_label} (confidence: {confidence_text})")
    elif predicted_class == expected_class:
        st.success(f"Prediction is correct: {predicted_label} (confidence: {confidence_text})")
    else:
        st.warning(
            "Model prediction does not match this reference image. "
            f"Expected {readable_label(expected_class)}, got {predicted_label} "
            f"(confidence: {confidence_text})."
        )


def main():
    st.set_page_config(page_title="Crop Disease Prediction", layout="centered")
    st.title("Crop Disease Prediction")

    project_root = Path(__file__).resolve().parents[1]
    tomato_classes = load_json_list(project_root / "models_tomato" / "classes.json")

    model, classes, error = load_artifacts(project_root)
    if model is None:
        st.error("Could not load the prediction model.")
        if error:
            st.caption(error)
        return

    if tomato_classes and classes != tomato_classes:
        st.warning("The loaded model classes do not match models_tomato/classes.json.")

    input_mode = st.radio("Input", ["Upload image", "Reference sample"], horizontal=True)
    image = None
    expected_class = None

    if input_mode == "Upload image":
        file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])
        if file is not None:
            image = Image.open(file).convert("RGB")
    else:
        sample_classes = [cls for cls in classes if cls in TOMATO_SAMPLE_IMAGES]
        selected = st.selectbox(
            "Reference sample",
            sample_classes,
            format_func=readable_label,
            label_visibility="collapsed",
        )
        expected_class = selected
        try:
            image_bytes = load_remote_image(TOMATO_SAMPLE_IMAGES[selected])
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            st.error(f"Could not load the reference image: {exc}")

    if image is None:
        return

    st.image(image, caption="Input image", use_container_width=True)
    results = predict(image, model, classes)
    show_prediction(results, expected_class=expected_class)


if __name__ == "__main__":
    main()
