"""Extended evaluation utility for CropDiseasePrediction.

Usage examples:
  python src/evaluate_metrics.py --model_path models/model.pt --classes models/classes.json --val_dir data/val --out_dir models/plots
  python src/evaluate_metrics.py --preds outputs/preds_run1.npz --classes models/classes.json --out_dir models/plots

This script will:
 - load predictions from a .npz or run the model on a validation folder
 - compute a wide set of metrics (per README suggestions)
 - save `models/metrics_runX.json`, plots to `out_dir`, and optional preds .npz
"""

import argparse
import json
from pathlib import Path
import numpy as np
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def safe_import_sklearn():
    try:
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            balanced_accuracy_score,
            matthews_corrcoef,
            cohen_kappa_score,
            roc_auc_score,
            average_precision_score,
            top_k_accuracy_score,
            log_loss,
            precision_recall_curve,
        )
        from sklearn.calibration import calibration_curve
        from sklearn.preprocessing import label_binarize
        return {
            'classification_report': classification_report,
            'confusion_matrix': confusion_matrix,
            'balanced_accuracy_score': balanced_accuracy_score,
            'matthews_corrcoef': matthews_corrcoef,
            'cohen_kappa_score': cohen_kappa_score,
            'roc_auc_score': roc_auc_score,
            'average_precision_score': average_precision_score,
            'top_k_accuracy_score': top_k_accuracy_score,
            'log_loss': log_loss,
            'precision_recall_curve': precision_recall_curve,
            'calibration_curve': calibration_curve,
            'label_binarize': label_binarize,
        }
    except Exception as e:
        print("scikit-learn is required for extended evaluation. Install it: pip install scikit-learn matplotlib numpy", file=sys.stderr)
        raise


def compute_ece(probs, labels, n_bins=10):
    # ECE: compare confidence to accuracy in bins using predicted class confidence
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accuracies = (preds == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_centers = []
    bin_acc = []
    bin_conf = []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1]) if i < n_bins - 1 else (confidences >= bins[i]) & (confidences <= bins[i + 1])
        if np.any(mask):
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            prop = mask.mean()
            ece += np.abs(conf - acc) * prop
            bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
            bin_acc.append(acc)
            bin_conf.append(conf)
    return ece, np.array(bin_centers), np.array(bin_acc), np.array(bin_conf)


def run_model_on_folder(model_path, classes_path, val_dir, batch_size=64, num_workers=0, device='cpu'):
    classes = json.loads(Path(classes_path).read_text(encoding='utf-8'))
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(str(val_dir), transform=tfms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device(device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.append(probs)
    y_prob = np.vstack(y_prob) if y_prob else np.zeros((len(y_true), len(classes)))
    return np.array(y_true), np.array(y_pred), y_prob, ds.classes


def save_json_metrics(out_path: Path, metrics: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')


def plot_and_save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches='tight', dpi=150)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Extended evaluation for CropDiseasePrediction")
    parser.add_argument('--model_path')
    parser.add_argument('--classes', required=True)
    parser.add_argument('--val_dir', default='data/val')
    parser.add_argument('--preds', help='Optional .npz with y_true,y_pred,y_prob')
    parser.add_argument('--out_dir', default='models/plots')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    skl = safe_import_sklearn()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.preds:
        data = np.load(args.preds)
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_prob = data['y_prob']
        classes = json.loads(Path(args.classes).read_text(encoding='utf-8'))
        ds_classes = classes
    else:
        if not args.model_path:
            raise ValueError('Either --preds or --model_path must be provided')
        y_true, y_pred, y_prob, ds_classes = run_model_on_folder(
            args.model_path, args.classes, args.val_dir, batch_size=args.batch_size, num_workers=args.num_workers, device=args.device
        )
        preds_out = Path('outputs')
        preds_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(preds_out / 'preds_run1.npz', y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    C = y_prob.shape[1]
    # classification report
    report = skl['classification_report'](y_true, y_pred, output_dict=True, zero_division=0)
    cm = skl['confusion_matrix'](y_true, y_pred)
    balanced = float(skl['balanced_accuracy_score'](y_true, y_pred))
    mcc = float(skl['matthews_corrcoef'](y_true, y_pred))
    kappa = float(skl['cohen_kappa_score'](y_true, y_pred))
    logloss = float(skl['log_loss'](y_true, y_prob))

    # ROC / PR
    y_onehot = skl['label_binarize'](y_true, classes=range(C))
    try:
        roc_ovr = float(skl['roc_auc_score'](y_onehot, y_prob, average='macro', multi_class='ovr'))
    except Exception:
        roc_ovr = None
    ap_per_class = []
    for k in range(C):
        try:
            ap = float(skl['average_precision_score'](y_onehot[:, k], y_prob[:, k]))
        except Exception:
            ap = None
        ap_per_class.append(ap)

    topk = float(skl['top_k_accuracy_score'](y_true, y_prob, k=args.top_k))

    # Brier per class
    from sklearn.metrics import brier_score_loss
    brier_per_class = []
    for k in range(C):
        brier_per_class.append(float(brier_score_loss((y_true == k).astype(int), y_prob[:, k])))

    # ECE
    ece, bin_centers, bin_acc, bin_conf = compute_ece(y_prob, y_true, n_bins=10)

    metrics = {
        'dataset': {
            'val_count': int(len(y_true)),
            'class_counts': {str(i): int((y_true == i).sum()) for i in range(C)}
        },
        'overall': {
            'accuracy': float(report.get('accuracy', 0.0)),
            'balanced_accuracy': balanced,
            'log_loss': logloss,
            'mcc': mcc,
            'cohen_kappa': kappa,
        },
        'averages': {
            'f1_macro': float(report.get('macro avg', {}).get('f1-score', 0.0)),
            'f1_weighted': float(report.get('weighted avg', {}).get('f1-score', 0.0)),
            'roc_auc_macro': roc_ovr,
            'pr_auc_macro': float(np.nanmean([a for a in ap_per_class if a is not None])) if any(a is not None for a in ap_per_class) else None,
            'top_{}_acc'.format(args.top_k): topk,
        },
        'per_class': {},
        'ece': float(ece),
    }

    for k in range(C):
        cls_name = ds_classes[k] if isinstance(ds_classes, (list, tuple)) and k < len(ds_classes) else str(k)
        metrics['per_class'][cls_name] = {
            'precision': float(report.get(str(k), {}).get('precision', 0.0)) if str(k) in report else float(report.get(cls_name, {}).get('precision', 0.0)),
            'recall': float(report.get(str(k), {}).get('recall', 0.0)) if str(k) in report else float(report.get(cls_name, {}).get('recall', 0.0)),
            'f1': float(report.get(str(k), {}).get('f1-score', 0.0)) if str(k) in report else float(report.get(cls_name, {}).get('f1-score', 0.0)),
            'support': int((y_true == k).sum()),
            'brier': float(brier_per_class[k]),
            'ap': float(ap_per_class[k]) if ap_per_class[k] is not None else None,
        }

    # Save metrics JSON
    metrics_path = Path('models') / 'metrics_run.json'
    save_json_metrics(metrics_path, metrics)

    # Save confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

        fig_cm, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ds_classes)
        disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
        fig_cm.suptitle('Confusion Matrix')
        plot_and_save(fig_cm, out_dir / 'confusion_matrix.png')

        # ROC curves per class
        fig_roc, ax = plt.subplots(figsize=(8, 6))
        for k in range(C):
            try:
                fpr, tpr, _ = roc_curve(y_onehot[:, k], y_prob[:, k])
                ax.plot(fpr, tpr, label=f"{ds_classes[k]} (AP={ap_per_class[k]:.2f})")
            except Exception:
                continue
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('Per-class ROC curves')
        ax.legend(loc='lower right', fontsize='small')
        plot_and_save(fig_roc, out_dir / 'roc_per_class.png')

        # Precision-Recall curves
        fig_pr, ax = plt.subplots(figsize=(8, 6))
        for k in range(C):
            try:
                precision, recall, _ = skl['precision_recall_curve'](y_onehot[:, k], y_prob[:, k])
                ax.plot(recall, precision, label=f"{ds_classes[k]}")
            except Exception:
                continue
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Per-class Precision-Recall curves')
        ax.legend(loc='lower left', fontsize='small')
        plot_and_save(fig_pr, out_dir / 'pr_per_class.png')

        # Calibration plot
        fig_cal, ax = plt.subplots(figsize=(6, 6))
        prob_true, prob_pred = skl['calibration_curve'](y_onehot.ravel(), y_prob.ravel(), n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Calibration curve (pooled)')
        plot_and_save(fig_cal, out_dir / 'calibration.png')
    except Exception as e:
        print('Plotting failed or matplotlib not available:', e)

    print(f"Saved metrics to: {metrics_path}")


if __name__ == '__main__':
    main()
