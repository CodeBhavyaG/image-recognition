# Animal Image Classifier — Capstone starter

Classifies animal photos using transfer learning (PyTorch + torchvision). It
maps Kaggle class folders (named in Italian) to English labels via a bilingual
vocabulary, trains a classifier, evaluates it, and emits *structured* JSON per
image in the capstone's Phase-2 format:

```json
{
  "subject": "red fox",
  "category": "animal",
  "attributes": ["domestic", "four-legged", "carnivore"],
  "caption": "A dog in a park.",
  "confidence": 0.9461
}
```

## Quick start

1. Drop your Kaggle data into `data/raw/image/<class>/`, one folder per class,
   named in Italian (folder names are auto-translated to English):

```
data/raw/image/
├── cane/        # -> "dog"
├── gatto/       # -> "cat"
├── cavallo/     # -> "horse"
└── ...
```

Nested wrappers are supported too (e.g. `data/raw/image/Animals-10/cane/...`).

2. Install dependencies (Python 3.12+):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

3. Train, evaluate, predict:

```bash
python main.py train                          # trains, saves data/models/best.pt
python main.py train --model mobilenet_v3_small --epochs 5 --freeze-backbone
python main.py evaluate                        # per-class report + top-1 precision + confusion matrix
python main.py predict --image path/to/img.jpg
python main.py predict --dir path/to/folder --topk 3
```

## Commands

| Command | What it does |
|---|---|
| `main.py train` | Trains the classifier on `data/raw/image/*`, saves best checkpoint |
| `main.py evaluate` | Prints per-class precision/recall/F1, **top-1 precision**, saves `data/outputs/confusion_matrix.png` + `metrics.json` |
| `main.py predict` | Classifies image(s) and prints validated structured JSON per image |

## Project layout

```
image-recognition/
├── config.py          # paths, hyperparameters, Italian->English vocabulary
├── schemas.py         # Pydantic ImageUnderstanding (the validated JSON schema)
├── class_metadata.py  # per-class attributes + caption template
├── producers.py       # classifier / vision-model producers + confidence guard
├── models.py          # transfer-learning model factory
├── dataset.py         # scans data/raw/image, stratified split, transforms
├── train.py           # training loop + best-checkpoint saving
├── evaluate.py        # report, top-1 precision, confusion matrix
├── predict.py         # classify an image -> validated JSON
├── main.py            # CLI entry point
└── data/
    ├── raw/image/<class>/   # <-- put the Kaggle dataset here
    ├── models/best.pt
    └── outputs/
```

## Reproducibility & the confidence guard

- A fixed `SEED` (config) gives a stratified, reproducible train/val split.
- Predictions below `CONFIDENCE_THRESHOLD` are **flagged, not silently accepted**
  (capstone "never trust unvalidated output" rule). Look for the `WARNING:` line
  from `main.py predict`.

## Honest limitations

The classifier only predicts the animal classes it was trained on; it cannot
truly "understand" arbitrary subjects/attributes. The JSON fields `subject` and
`confidence` come from the softmax head, while `attributes`/`caption` come from
curated class metadata. This gives you rich structured output today and a clean
seam to plug in a real vision model (Gemini Flash / Ollama) later — swap the
`Producer` in `producers.py` behind the same `schemas.ImageUnderstanding`.

## Reporting the headline metric

`main.py evaluate` prints the **top-1 precision** on the held-out validation
set — the capstone's headline quality number. Paste the printed value into your
`README` / `EVIDENCE.md` once you have a trained model.
