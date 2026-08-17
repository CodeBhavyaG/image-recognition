"""Central configuration: paths, model/hyperparameters, and the class vocabulary.

The Kaggle dataset folders are named in Italian (your ``translate`` vocabulary),
so raw class folders are mapped to English labels via IT_TO_EN below.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGE_DIR = DATA_DIR / "raw" / "image"   # each subfolder = one class (Italian name)
MODELS_DIR = DATA_DIR / "models"             # saved checkpoints
OUTPUTS_DIR = DATA_DIR / "outputs"           # eval artifacts (reports, confusion matrix)

# ---------------------------------------------------------------------------
# Class vocabulary  (Italian -> English)
# ---------------------------------------------------------------------------
IT_TO_EN = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider",
}

# Inverse map for convenient English -> Italian display.
EN_TO_IT = {v: k for k, v in IT_TO_EN.items()}

# All English class labels, sorted for reproducibility.
CLASS_NAMES = sorted(EN_TO_IT)

# ---------------------------------------------------------------------------
# Training / data
# ---------------------------------------------------------------------------
MODEL_NAME = "resnet18"      # "resnet18" | "mobilenet_v3_small"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
FREEZE_BACKBONE = False
WEIGHT_DECAY = 1e-4
SEED = 42
VAL_SPLIT = 0.2              # fraction of each class held out for validation
NUM_WORKERS = 0              # cross-platform-safe default
SAVE_BEST = True

# ---------------------------------------------------------------------------
# Confidence / guard
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.6   # predictions below this are flagged, not trusted