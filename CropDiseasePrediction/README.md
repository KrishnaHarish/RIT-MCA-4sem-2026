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
Use an image dataset organized like this (compatible with `torchvision.datasets.ImageFolder`):

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

If your dataset has a different structure (for example CSV labels or a single folder), update `src/train.py` accordingly.

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
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Train
```powershell
python .\src\train.py --data_dir .\data --epochs 10 --batch_size 32 --arch resnet18
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

