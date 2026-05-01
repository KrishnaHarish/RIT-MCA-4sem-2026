# Crop Disease Prediction (MCA - 4th Sem) — Ramaiah Institute of Technology

## Overview
This project predicts the crop disease class from an input plant image. It is designed for an MCA 4th semester academic submission and demonstrates applied ML (image classification) with a practical inference UI.

## Problem Statement
Farmers and agronomists often struggle to identify crop diseases early. An automated model that classifies disease from leaf/plant images can help recommend timely action and reduce yield loss.

## Objectives
- Build an image classification model to predict crop disease categories.
- Evaluate performance using standard metrics (accuracy, precision, recall, F1-score, confusion matrix).
- Provide a simple inference pipeline and UI for end-user testing.

## Dataset (expected format)
You can use either of these layouts (both are compatible with `torchvision.datasets.ImageFolder`):

```text
CropDiseasePrediction/
  data/
    train/
      class_1/
      class_2/
      ...
    val/
      class_1/
      class_2/
      ...
```

PlantVillage-style (common Kaggle/Korn dataset layout):

```text
CropDiseasePrediction/
  data/
    PlantVillage/
      class_1/
      class_2/
      ...
```

If you use PlantVillage-style (no `train/` and `val/`), the trainer will automatically split into train/val using `--val_ratio`.

## Methodology (typical)
1. Data loading and augmentation (train-time transforms).
2. Transfer learning (e.g., `ResNet18`) or training a custom CNN.
3. Training loop with cross-entropy loss.
4. Save best model and class names.
5. Inference: load model, predict top class, show probability.

## Evaluation
Recommended metrics:
- Accuracy
- Precision / Recall / F1-score (macro or weighted)
- Confusion matrix

## Deliverables
- Trained model artifact(s) in `models/`
- Inference script (`src/predict.py`)
- Optional Streamlit demo UI (`app/app.py`)

## Project Structure
```text
CropDiseasePrediction/
  app/
    app.py
  data/
    train/
    val/
  models/
    model.pt
    classes.json
  notebooks/
  src/
    train.py
    predict.py
  requirements.txt
  README.md
  .gitignore
```

## Setup
1. Create/activate your virtual environment:
   ```powershell
   ..\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

Run the above steps from the `CropDiseasePrediction` folder (so `..\.venv\...` resolves correctly), or from the repo root using `.\.venv\Scripts\Activate.ps1`.

## Download PlantVillage (No Kaggle - Zenodo)
Use the PlantVillage archive from Zenodo (no Kaggle credentials).

This workflow produces an `ImageFolder`-compatible layout:
- `data/train/<class>/...`
- `data/val/<class>/...`

### 1) Download the `.7z`
Download `plantvillage_deeplearning_paper_dataset.7z` from:
https://zenodo.org/records/1204914

Put it here (folder will be created):
```powershell
.\data\plantvillage_download\
```

### 2) Extract with 7-Zip
```powershell
"C:\Program Files\7-Zip\7z.exe" x .\data\plantvillage_download\plantvillage_deeplearning_paper_dataset.7z -o.\data\plantvillage_extracted -y
```

### 3) Export `train/val` folders for PyTorch
This uses the dataset’s provided split (`80-20/`) and copies images into `data/train` and `data/val`.

```powershell
python .\src\export_plantvillage_imagefolder.py --out_dir .\data --split_name 80-20 --link_mode hardlink
```

Optional: faster subset export for quick testing:
```powershell
python .\src\export_plantvillage_imagefolder.py --out_dir .\data --split_name 80-20 --link_mode copy --max_train_images 2000 --max_val_images 500
```

## Train
```powershell
python .\src\train.py --data_dir .\data --epochs 10 --batch_size 32 --arch resnet18
```

If your dataset is a single `ImageFolder` root (class subfolders directly under `--data_dir`, and no `train/`/`val/` folders yet), the trainer can auto-split into train/val:

```powershell
python .\src\train.py --data_dir .\data\<your_image_root> --epochs 10 --batch_size 32 --arch resnet18 --val_ratio 0.2 --seed 42
```
After training, artifacts will be written to:
- `models/model.pt`
- `models/classes.json`

## Predict
```powershell
python .\src\predict.py --model_path .\models\model.pt --classes_path .\models\classes.json --image_path "path_to_image.jpg"
```

## Run UI (Streamlit)
```powershell
streamlit run .\app\app.py
```

## Notes
- Training time depends on dataset size and hardware (GPU speeds it up).
- If you want your project to match a specific dataset (PlantVillage, Kaggle, etc.), tell me the source and expected folder layout, and I’ll tailor `train.py` + README accordingly.

## Quick Demo (no heavy deps)

To try a dependency-free smoke demo that does not require PyTorch or Streamlit:

1. Extract the bundled sample image:

```bash
python extract_sample_image.py
```

2. Run the lightweight smoke predictor:

```bash
python run_smoke_predict.py --image models/sample.jpg
```

This prints a deterministic pseudo-prediction using `models/classes.json`.

Full app (Streamlit)

To run the real Streamlit UI (requires dependencies / PyTorch):

```bash
# optional venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Docker (isolated)

You can build and run the included `Dockerfile` to avoid local dependency issues:

```bash
docker build -t crop-disease-app -f Dockerfile .
docker run --rm -p 8501:8501 -v "$(pwd)/models":/app/models crop-disease-app
```

If you want GPU support, ask me to add a CUDA-capable Dockerfile variant.

