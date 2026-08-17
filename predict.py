"""Prediction: classify an image (or every image in a folder) and emit a
*validated* `ImageUnderstanding` JSON object per image — matching the capstone
Phase-2 structured output.
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from config import MODELS_DIR, IMAGE_SIZE, EN_TO_IT, CONFIDENCE_THRESHOLD
from models import get_transfer_model, IMAGE_NET_MEAN, IMAGE_NET_STD
from producers import ClassifierProducer, flag_low_confidence
from utils import resolve_device


def _preprocess(size=IMAGE_SIZE):
    resize = int(size * 1.14)
    return T.Compose([
        T.Resize((resize, resize)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
    ])


def load_checkpoint(checkpoint: Path, device):
    ckpt = torch.load(checkpoint, map_location=device)
    model, _ = get_transfer_model(ckpt["model_name"], num_classes=len(ckpt["class_names"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["class_names"]


def predict_topk(model, class_names, image_path: Path, device, topk: int = 1):
    img = Image.open(image_path).convert("RGB")
    x = _preprocess()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    scores, idxs = torch.topk(probs, k=min(topk, len(class_names)))
    return [(class_names[i], float(s)) for i, s in zip(idxs.tolist(), scores.tolist())]


def _emit(understanding, file=None):
    payload = understanding.model_dump()  # {"subject","category","attributes","caption","confidence"}
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=file)


def predict_main(checkpoint, source, topk=1, threshold=CONFIDENCE_THRESHOLD):
    checkpoint = Path(checkpoint) if checkpoint else MODELS_DIR / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}. Train first: `python main.py train`.")

    source = Path(source)
    images = [source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ) if source.is_dir() else []
    if not images:
        raise FileNotFoundError(f"No images found at {source!s}")

    device = resolve_device()
    model, class_names = load_checkpoint(checkpoint, device)
    producer = ClassifierProducer()

    for path in images:
        top = predict_topk(model, class_names, path, device, topk=topk)
        subject, confidence = top[0]
        understanding = producer.produce(subject, confidence)
        italian = EN_TO_IT.get(subject)

        print(f"// {path}")
        _emit(understanding)

        flag = flag_low_confidence(understanding)
        if flag:
            print(f"// WARNING: {flag}", file=sys.stderr)
        if topk > 1:
            ranked = [
                {"rank": r + 1, "subject": s, "confidence": round(c, 4)}
                for r, (s, c) in enumerate(top)
            ]
            print(f"// top-{topk}: {json.dumps(ranked)}")
        if italian:
            print(f"// italian: {italian}")
        print()

    return 0