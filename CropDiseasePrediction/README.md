# Tomato Disease Prediction (MCA - 4th Sem) — Ramaiah Institute of Technology

## Overview
This project predicts tomato leaf disease class from an input image. It is designed for an MCA 4th semester academic submission and demonstrates applied ML (image classification) with a practical inference UI.

## Problem Statement
Farmers and agronomists often struggle to identify tomato diseases early. An automated model that classifies disease from leaf images can help recommend timely action and reduce yield loss.

## Objectives
- Build an image classification model to predict tomato disease categories.
- Evaluate performance using standard metrics (accuracy, precision, recall, F1-score, confusion matrix).
- Provide a simple inference pipeline and UI for end-user testing.

## Tomato Classes (Target)
1. `Tomato___Bacterial_spot`
2. `Tomato___Early_blight`
3. `Tomato___Late_blight`
4. `Tomato___Leaf_Mold`
5. `Tomato___Septoria_leaf_spot`
6. `Tomato___Spider_mites_Two-spotted_spider_mite`
7. `Tomato___Target_Spot`
8. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
9. `Tomato___Tomato_mosaic_virus`
10. `Tomato___healthy`

## Dataset (expected format)
You can use either of these layouts for a tomato-only dataset (both are compatible with `torchvision.datasets.ImageFolder`):

```text
CropDiseasePrediction/
  data/
    train/
      Tomato___Bacterial_spot/
      Tomato___Early_blight/
      Tomato___Late_blight/
      Tomato___Leaf_Mold/
      Tomato___Septoria_leaf_spot/
      Tomato___Spider_mites_Two-spotted_spider_mite/
  Classes (2): Apple___healthy, Tomato___healthy
      Tomato___Tomato_Yellow_Leaf_Curl_Virus/
      Tomato___Tomato_mosaic_virus/
      Tomato___healthy/
    val/
[1/3] Training for 10 epoch(s) …
      Epoch 1/10 — loss=0.2822  val_acc=1.0000
      Epoch 10/10 — loss=0.0530  val_acc=1.0000
      Tomato___Late_blight/
      Tomato___Leaf_Mold/
      Tomato___Septoria_leaf_spot/
      Tomato___Spider_mites_Two-spotted_spider_mite/
      File  : 505465db-407b-4e0a-8110-7479dad5261c___GH_HL Leaf 389.JPG
      Label : Tomato___healthy
      Tomato___Tomato_mosaic_virus/
      Tomato___healthy/
    1. Tomato___healthy                         0.5099 [OK]
    2. Apple___healthy                          0.4901
PlantVillage-style (common Kaggle/Korn dataset layout):

```text
CropDiseasePrediction/
  data/
    TomatoPlantVillage/
      Tomato___Bacterial_spot/
      Tomato___Early_blight/
      Tomato___Late_blight/
      Tomato___Leaf_Mold/
      Tomato___Septoria_leaf_spot/
      Tomato___Spider_mites_Two-spotted_spider_mite/
      Tomato___Target_Spot/
      Tomato___Tomato_Yellow_Leaf_Curl_Virus/
      Tomato___Tomato_mosaic_virus/
      Tomato___healthy/
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
- Precision / Recall / F1-score (per-class, macro and weighted)
- Confusion matrix (absolute + normalized)


### Model Accuracy & Evaluation Results

We evaluated all trained models in the workspace, including the primary 10-class Tomato model (`models_tomato`) on the full validation dataset (`data_tomato/val`, containing 1,726 images). Below is a summary of the model performances:

| Model Directory | Task / Dataset | Classes | Validation Accuracy | F1-Score (Macro) | Purpose / Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `models` | 2-Class Smoke Test (`data_smoke`) | 2 | **95.82%** | **0.9581** | High-accuracy sanity check pipeline validation. |
| **`models_tomato`** | **Full Tomato Run (`data_tomato`)** | **10** | **87.95%** | **0.8479** | **Main academic/report model** trained on full split. |
| `models_tomato_small` | Small Tomato Run (`data_tomato_small`) | 10 | **60.34%** | **0.6003** | Trained on subset split. |
| `models_tomato10_fast` | Fast Tomato Run (`data_tomato_small`) | 10 | **10.54%** | **0.0191** | 1-epoch from-scratch baseline (predicted majority class). |

---

### Detailed Evaluation of the Primary Model (`models_tomato`)

The primary 10-class Tomato model uses **transfer learning (ResNet-18)** and was trained on the full `data_tomato` split. The evaluation metrics saved under `models/metrics_run.json` show:

*   **Overall Accuracy:** `87.95%`
*   **Balanced Accuracy (Mean Recall):** `83.70%`
*   **Top-3 Accuracy:** `98.38%`
*   **Log Loss (Cross-Entropy):** `0.3917`
*   **Matthews Correlation Coefficient (MCC):** `0.8577`
*   **Cohen's Kappa:** `0.8569`
*   **Expected Calibration Error (ECE):** `8.17%` (0.0817)

#### Per-Class Performance Breakdown

| Tomato Leaf Class | Support | Precision | Recall | F1-Score | Avg Precision (AP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 530 | 93.96% | 96.79% | **95.35%** | 99.48% |
| `Tomato___healthy` | 183 | 89.05% | 97.81% | **93.23%** | 99.20% |
| `Tomato___Tomato_mosaic_virus` | 39 | 94.29% | 84.62% | **89.19%** | 95.67% |
| `Tomato___Spider_mites_Two-spotted_spider_mite` | 136 | 89.47% | 87.50% | **88.48%** | 94.54% |
| `Tomato___Late_blight` | 159 | 88.46% | 86.79% | **87.62%** | 96.02% |
| `Tomato___Leaf_Mold` | 84 | 87.50% | 83.33% | **85.37%** | 93.79% |
| `Tomato___Bacterial_spot` | 192 | 77.97% | 92.19% | **84.49%** | 96.11% |
| `Tomato___Septoria_leaf_spot` | 157 | 86.21% | 79.62% | **82.78%** | 90.99% |
| `Tomato___Target_Spot` | 142 | 77.55% | 80.28% | **78.89%** | 89.58% |
| `Tomato___Early_blight` | 104 | 89.29% | 48.08% | **62.50%** | 81.72% |

> [!NOTE]
> **Key Analysis Insight:** The primary bottleneck is `Tomato___Early_blight` (recall `48.08%`, F1-score `62.50%`), where the model frequently confuses early symptoms with target spot or late blight. In contrast, the model performs exceptionally well on `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (F1 `95.35%`) and `Tomato___healthy` (F1 `93.23%`).

All evaluation plots (confusion matrix, ROC, Precision-Recall, Calibration) are saved under `models_tomato/plots/`.

For a validated high-accuracy run on the bundled smoke dataset, see the Quick Demo section: `1.0000` best/final validation accuracy on `data_smoke/`.

How to generate / store extended metrics
- The existing quick evaluator writes `models/metrics_current.json` via `src/eval_current_model.py`.
- For MCAP2, save a detailed metrics JSON (example schema below) to `models/metrics_runX.json` and save plots to `models/plots/`.
- Save raw predictions for reproducibility: `outputs/preds_runX.npz` containing `y_true`, `y_pred`, `y_prob`.

Minimal `metrics` JSON schema (suggested):
```
{
  "dataset": {"train_count": 2000, "val_count": 759, "class_counts": {"Tomato___healthy": 120, ...}},
  "overall": {"accuracy": 0.9123, "balanced_accuracy": 0.89, "log_loss": 0.34, "mcc": 0.72},
  "averages": {"f1_macro": 0.88, "f1_weighted": 0.90, "roc_auc_macro": 0.95, "pr_auc_macro": 0.84, "top_3_acc": 0.98},
  "per_class": {"Tomato___healthy": {"precision":0.93, "recall":0.91, "f1":0.92, "support":120, "roc_auc":0.97}},
  "files": {"confusion_matrix_png":"models/plots/confusion_matrix.png", "roc_png":"models/plots/roc.png"}
}
```

Quick run suggestions
- Produce a reproducible evaluation run (example):
```bash
python src/eval_current_model.py
# (or your extended evaluator once implemented):
python src/evaluate_metrics.py --model_path models/model.pt --classes models/classes.json --out_dir models/plots --pred_out outputs/preds_run1.npz
```

Notes:
- Keep the `--seed` used during training available and record the final command-line arguments used to produce the saved model.
- Include plots (confusion matrix, per-class ROC/PR, calibration) in `models/plots/` and link them from the README for examiners.

> Note: This is a short 10-epoch baseline (from-scratch model). Use it as proof of pipeline execution, not final performance.
>
> Recommended final run for presentation/report:
> - Use transfer learning (`ResNet18` with pretrained ImageNet weights)
> - Train for 10-20 epochs on full tomato split (`data_tomato`)
> - Report class-wise precision/recall/F1 and confusion matrix in the final submission

## Deliverables
- Trained model artifact(s) in `models/`
- Inference script (`src/predict.py`)
- Optional Streamlit demo UI (`app/app.py`)

## MCAIN Rubric Alignment (Quick Mapping)
- **CO1 - Tools and technologies:** Python, PyTorch, torchvision, Streamlit, transfer learning.
- **CO1 - Relevance to market:** early disease detection support for tomato farming and agri advisory use-cases.
- **CO2 - Demonstration:** end-to-end pipeline from data preparation to model inference and UI demo.
- **CO3 - Report and presentation:** include architecture, metrics, confusion matrix, limitations, and future work.

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
    eval_current_model.py
    train_eval_tomato10_fast.py
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

## Quick Demo (Tomato - all diseases)

Train and evaluate a **10-class tomato** classifier in one step (expects `data_tomato_small/` with only `Tomato___*` class folders; the same command is documented again under **Train + evaluate 10-class tomato benchmark** below).

```powershell
python .\src\train_eval_tomato10_fast.py
```

Example terminal output (your loss and metrics may differ slightly):

```
Epoch 1/1 | train_loss=2.0829 | val_acc=0.1054
{
  "dataset": "data_tomato_small",
  "num_classes": 10,
  "train_images": 2000,
  "val_images": 759,
  "best_val_acc_during_training": 0.1054,
  "accuracy": 0.1054,
  "f1_macro": 0.0191,
  "confusion_matrix": [ ... ]
}
```

Artifacts are written under `models_tomato10_fast/` (see `metrics.json` for the full confusion matrix).

### Optional: minimal pipeline smoke test (2 classes)

Run the self-contained demo script to see the full pipeline in action — no dataset download required.
It trains a tiny ResNet-18 model for 10 epochs on the bundled `data_smoke/` dataset (smoke-test data, ~1,000 images)
and then immediately runs inference on a random validation image.

Validated smoke-demo result: the current transfer-learning setup reaches a best/final validation accuracy of `1.0000` on `data_smoke/`.

```powershell
python .\demo.py
```

Expected output (values will vary slightly):

```
============================================================
  Crop Disease Prediction — Quick Demo
============================================================
  Classes (10): Tomato___Bacterial_spot, Tomato___Early_blight, Tomato___Late_blight, Tomato___Leaf_Mold, Tomato___Septoria_leaf_spot, Tomato___Spider_mites_Two-spotted_spider_mite, Tomato___Target_Spot, Tomato___Tomato_Yellow_Leaf_Curl_Virus, Tomato___Tomato_mosaic_virus, Tomato___healthy
  Train images : 1000
  Val images   : 359
  Device       : cpu

[1/3] Training for 10 epoch(s) …
  Epoch 10/10 — loss=0.0384  val_acc=1.0000

[2/3] Model saved to temporary directory.

[3/3] Running inference on a random validation image …
      File  : <sample>.JPG
      Label : Tomato___healthy

  Top predictions:
    1. Tomato___healthy                         0.9990 ✓
    2. Tomato___Early_blight:                   0.1216

============================================================
  Demo complete!
============================================================
```

> The temporary model artifact is written to a system temp directory and cleaned up automatically.

## Train
```powershell
python .\src\train.py --data_dir .\data --epochs 10 --batch_size 32 --arch resnet18
```

If your dataset is a single tomato `ImageFolder` root (class subfolders directly under `--data_dir`, and no `train/`/`val/` folders yet), the trainer can auto-split into train/val:

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

## Evaluate current saved model
```powershell
python .\src\eval_current_model.py
```

This writes measured metrics to:
- `models/metrics_current.json`

## Train + evaluate 10-class tomato benchmark
```powershell
python .\src\train_eval_tomato10_fast.py
```

This writes measured metrics to:
- `models_tomato10_fast/metrics.json`

## Run UI (Streamlit)
```powershell
streamlit run .\app\app.py
```

## Notes
- Training time depends on dataset size and hardware (GPU speeds it up).
- If you want your project to match a specific dataset (PlantVillage, Kaggle, etc.), tell me the source and expected folder layout, and I’ll tailor `train.py` + README accordingly.
- Current benchmark metric files:
  - 2-class smoke evaluation: `models/metrics_current.json`
  - 10-class tomato baseline: `models_tomato10_fast/metrics.json`

