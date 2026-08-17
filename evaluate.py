"""Evaluation: per-class report, confusion-matrix PNG, and the capstone's
headline "top-1 precision" number (paste-able into README / EVIDENCE.md).

Metrics are computed with plain NumPy so evaluation has no hard dependency on
scikit-learn. If scikit-learn happens to be installed, its richer
``classification_report`` is additionally printed for convenience.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    RAW_IMAGE_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, SEED,
)
from dataset import build_datasets
from models import get_transfer_model
from utils import resolve_device


def _load_model(checkpoint: Path, device, fallback_classes):
    ckpt = torch.load(checkpoint, map_location=device)
    class_names = ckpt.get("class_names") or fallback_classes
    model, _ = get_transfer_model(ckpt["model_name"], num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def evaluate(checkpoint=None, raw_image_dir=RAW_IMAGE_DIR, output_dir=OUTPUTS_DIR):
    checkpoint = Path(checkpoint) if checkpoint else MODELS_DIR / "best.pt"
    output_dir = Path(output_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. Train first: `python main.py train`."
        )

    device = resolve_device()
    _, val_ds, class_names = build_datasets(raw_image_dir, image_size=IMAGE_SIZE, seed=SEED)
    model, class_names = _load_model(checkpoint, device, class_names)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    n = len(all_labels)
    n_classes = len(class_names)

    # Top-1 precision (== accuracy for single-label tasks) — the headline metric.
    top1 = float((all_preds == all_labels).mean())

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    # Per-class precision / recall / f1 / support in pure NumPy.
    metrics = []
    for c in range(n_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        support = float(cm[c, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics.append((prec, rec, f1, support))

    macro = np.mean([m[:3] for m in metrics], axis=0)
    total_support = sum(m[3] for m in metrics)
    weighted = (
        sum(m[0] * m[3] for m in metrics) / total_support if total_support else 0.0,
        sum(m[1] * m[3] for m in metrics) / total_support if total_support else 0.0,
        sum(m[2] * m[3] for m in metrics) / total_support if total_support else 0.0,
    )

    header = f"{'class':<12}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}"
    lines = [header, "-" * len(header)]
    for i, (p, r, f, s) in enumerate(metrics):
        lines.append(f"{class_names[i]:<12}{p:>10.3f}{r:>9.3f}{f:>8.3f}{s:>9.0f}")

    print("=" * 64)
    print(f"Model head labels ({n_classes} classes): {class_names}")
    print(f"Validation samples: {n}")
    print(f"Top-1 precision (== accuracy for single-label tasks): {top1:.4f}")
    print("=" * 64)
    print("\n".join(lines))
    print(f"\naccuracy        : {top1:.4f}")
    print(f"macro avg       : {macro[0]:.3f}  {macro[1]:.3f}  {macro[2]:.3f}")
    print(f"weighted avg    : {weighted[0]:.3f}  {weighted[1]:.3f}  {weighted[2]:.3f}")

    # Optional richer report when scikit-learn is available.
    try:
        from sklearn.metrics import classification_report
        print("\n(scikit-learn classification_report:)")
        print(classification_report(all_labels, all_preds, labels=list(range(n_classes)),
                                    target_names=class_names, zero_division=0))
    except Exception:
        print("\n(scikit-learn not available; NumPy-based report above is authoritative.)")


    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, n_classes), max(6, n_classes)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n_classes), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (val)")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved to: {cm_path}")

    (output_dir / "report.txt").write_text("\n".join(lines) + "\n")
    (output_dir / "metrics.json").write_text(
        __import__("json").dumps(
            {
                "top1_precision": top1,
                "accuracy": top1,
                "num_val_samples": n,
                "classes": class_names,
                "macro_p_r_f": [round(float(x), 4) for x in macro],
                "weighted_p_r_f": [round(float(x), 4) for x in weighted],
                "per_class": {c: [round(float(x), 4) for x in m] for c, m in zip(class_names, metrics)},
            },
            indent=2,
        )
    )
    print(f"metrics.json and report.txt saved to: {output_dir}")
    return top1