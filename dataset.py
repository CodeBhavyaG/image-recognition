"""Dataset + transforms for the `data/raw/image/<class>/` Kaggle layout.

Class folders are matched against the Italian -> English vocabulary in config,
so `data/raw/image/cane/*.jpg` is labelled as class "dog".
"""
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from config import IT_TO_EN, VAL_SPLIT, SEED, IMAGE_SIZE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]


def _has_images(d: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in d.iterdir())


def normalize_label(folder_name: str) -> str:
    """Map a raw folder name (Italian, or already English) to an English label."""
    key = folder_name.strip().lower()
    return IT_TO_EN.get(key, folder_name.strip())


def find_image_root(base_dir) -> Path:
    """Return the directory whose subfolders are image classes.

    Tolerates a nested wrapper folder, e.g. ``raw/image/Animals-10/cane/...``.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Direct layout: base/<class>/<image>
    if any(p.is_dir() and _has_images(p) for p in base.iterdir()):
        return base

    # Nested layout: base/<wrapper>/<class>/<image> — pick the first wrapper.
    for child in base.iterdir():
        if child.is_dir() and any(p.is_dir() and _has_images(p) for p in child.iterdir()):
            return child

    raise FileNotFoundError(
        f"No image class folders found under {base}. Put images as "
        f"`<root>/<class>/<image.jpg>` (e.g. `.../cane/dog_001.jpg`)."
    )


class AnimalDataset(Dataset):
    """Torch Dataset over (image_path, label_index) samples."""

    def __init__(self, samples: List[Tuple[Path, int]], transform=None, image_size: int = IMAGE_SIZE):
        self.samples = samples
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img) if self.transform else T.ToTensor()(img)
        except Exception:
            # Corrupt/unreadable image -> black placeholder so the batch stays aligned.
            img = torch.zeros(3, self.image_size, self.image_size)
        return img, label


def _train_transform(size: int):
    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
        ]
    )


def _eval_transform(size: int):
    resize = int(size * 1.14)
    return T.Compose(
        [
            T.Resize((resize, resize)),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
        ]
    )


def build_datasets(
    raw_image_dir=...,
    image_size: int = IMAGE_SIZE,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
):
    """Scan ``data/raw/image`` and return (train_ds, val_ds, english_class_names).

    The split is stratified per class for reproducible results.
    """
    root = find_image_root(raw_image_dir)
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir() and _has_images(p))
    if not class_dirs:
        raise FileNotFoundError(f"No class folders with images found under {root}.")

    # first-seen-order of unique English labels (stable, reproducible)
    english_labels: List[str] = []
    for p in class_dirs:
        en = normalize_label(p.name)
        if en not in english_labels:
            english_labels.append(en)
    label_index = {en: i for i, en in enumerate(english_labels)}

    samples: List[Tuple[Path, int]] = []
    for p in class_dirs:
        en = normalize_label(p.name)
        idx = label_index[en]
        for img in sorted(p.iterdir()):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((img, idx))

    if not samples:
        raise FileNotFoundError(f"No images found under {root}.")

    by_label: dict = defaultdict(list)
    for path, idx in samples:
        by_label[idx].append((path, idx))

    import random

    rng = random.Random(seed)
    train, val = [], []
    for idx, lst in by_label.items():
        rng.shuffle(lst)
        n_val = max(1, int(len(lst) * val_split)) if len(lst) > 1 else 0
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])

    return (
        AnimalDataset(train, transform=_train_transform(image_size), image_size=image_size),
        AnimalDataset(val, transform=_eval_transform(image_size), image_size=image_size),
        english_labels,
    )