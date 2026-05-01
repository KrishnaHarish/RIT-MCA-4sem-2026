"""Smoke tests for the Crop Disease Prediction pipeline."""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import get_model  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DUMMY_CLASSES = ["Apple___healthy", "Tomato___healthy"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_model_dir(tmp_path):
    """Create a temp directory with a tiny random-weight model and classes.json."""
    model = get_model("resnet18", num_classes=len(DUMMY_CLASSES))
    model.eval()

    model_path = tmp_path / "model.pt"
    classes_path = tmp_path / "classes.json"

    torch.save(model.state_dict(), model_path)
    classes_path.write_text(json.dumps(DUMMY_CLASSES, indent=2), encoding="utf-8")

    return tmp_path


@pytest.fixture
def dummy_image(tmp_path):
    """Create a tiny solid-color RGB image for testing."""
    img = Image.new("RGB", (256, 256), color=(120, 180, 80))
    image_path = tmp_path / "leaf.jpg"
    img.save(image_path)
    return image_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_model_resnet18():
    model = get_model("resnet18", num_classes=5)
    assert model.fc.out_features == 5


def test_get_model_resnet50():
    model = get_model("resnet50", num_classes=10)
    assert model.fc.out_features == 10


def test_model_forward_pass():
    """Model should produce logits of shape [batch, num_classes]."""
    num_classes = 4
    model = get_model("resnet18", num_classes=num_classes)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, num_classes)


def test_predict_pipeline(dummy_model_dir, dummy_image):
    """End-to-end prediction pipeline: load model -> transform image -> infer."""
    classes = json.loads((dummy_model_dir / "classes.json").read_text(encoding="utf-8"))
    num_classes = len(classes)

    device = torch.device("cpu")
    model = get_model("resnet18", num_classes=num_classes).to(device)
    state_dict = torch.load(dummy_model_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    img = Image.open(dummy_image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    assert probs.shape == (num_classes,)
    assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities should sum to 1"

    top_prob, top_idx = torch.max(probs, dim=0)
    predicted_class = classes[top_idx.item()]
    assert predicted_class in DUMMY_CLASSES
    assert 0.0 < top_prob.item() <= 1.0


def test_create_demo_model():
    """create_demo_model.py main() should produce valid model and classes artifacts."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        result = subprocess.run(
            [sys.executable, str(SRC_DIR / "create_demo_model.py"), "--output_dir", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"create_demo_model.py failed:\n{result.stderr}"

        model_path = out / "model.pt"
        classes_path = out / "classes.json"
        assert model_path.exists(), "model.pt not created"
        assert classes_path.exists(), "classes.json not created"

        classes = json.loads(classes_path.read_text(encoding="utf-8"))
        assert len(classes) > 0

        # Verify the saved model can be loaded and used for inference
        from create_demo_model import DEFAULT_CLASSES, get_model as demo_get_model

        assert len(classes) == len(DEFAULT_CLASSES)
        model = demo_get_model("resnet18", num_classes=len(classes))
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out_tensor = model(x)
        assert out_tensor.shape == (1, len(classes))
