"""Training loop for the transfer-learning animal classifier."""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    MODEL_NAME, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    FREEZE_BACKBONE, NUM_WORKERS, SAVE_BEST, MODELS_DIR, SEED, RAW_IMAGE_DIR,
)
from dataset import build_datasets
from models import get_transfer_model
from utils import resolve_device, set_seed


def _correct(logits, labels) -> int:
    return int((logits.argmax(dim=1) == labels).sum().item())


def train_all(
    model_name=MODEL_NAME,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    freeze_backbone=FREEZE_BACKBONE,
    num_workers=NUM_WORKERS,
    save_best=SAVE_BEST,
    seed=SEED,
    raw_image_dir=RAW_IMAGE_DIR,
):
    set_seed(seed)
    device = resolve_device()
    raw_image_dir = Path(raw_image_dir)
    print(f"PyTorch {torch.__version__} on device: {device}\n")

    train_ds, val_ds, class_names = build_datasets(raw_image_dir, image_size=image_size, seed=seed)
    print(f"Class labels (English): {class_names}")
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model, _ = get_transfer_model(model_name, num_classes=len(class_names), freeze_backbone=freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / "best.pt"
    best_val_acc = -1.0  # start below 0 so the first epoch is always saved

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_correct += _correct(logits, labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running_loss / max(len(train_ds), 1)

        # validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, len(val_ds)
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device, non_blocking=True)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * images.size(0)
                val_correct += _correct(logits, labels)
        val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        tag = ""
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "class_names": class_names,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
            tag = "  (best model saved)"

        print(f"[epoch {epoch:>3}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{tag}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    if checkpoint_path.exists():
        print(f"Checkpoint saved to: {checkpoint_path}")
    else:
        print("No checkpoint saved (validation accuracy never improved).")
    return class_names