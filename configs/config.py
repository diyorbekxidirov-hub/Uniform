"""
configs/config.py  —  Uniform3 single source of truth.
All 6 datasets. Weights: D1=38% D2=12% D3=2% D4=8% D5=18% D6=22%
"""
from pathlib import Path
from typing import List

BASE = Path("/home/dior/Projects/BigDataset")

DATASET_ROOTS = {
    1: BASE / "Dataset_1",
    2: BASE / "Dataset_2",
    3: BASE / "Dataset_3",
    4: BASE / "Dataset_4",
    5: BASE / "Dataset_5",
    6: BASE / "Dataset_6",
}

PROJECT_ROOT   = Path("/home/dior/Projects/Uniform3")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "logs"
SAMPLES_DIR    = PROJECT_ROOT / "samples"

# Weights sum = 1.0  (D1=38 D2=12 D3=2 D4=8 D5=18 D6=22)
DATASET_WEIGHTS: List[float] = [0.38, 0.12, 0.02, 0.08, 0.18, 0.22]

# Dataset_1
D1_MIN_IMAGES_PER_CLASS = 2
D1_STRICT_SAME_PLAY     = True

# Dataset_4
D4_CLASSES = ["A", "B", "C", "D"]

# Dataset_5
D5_ANCHOR_CLASSES  = ["A", "B", "C"]
D5_NEG_ONLY        = ["D", "X"]
D5_B_OVERSAMPLE    = True
D5_XCLASS_NEG_PROB = 0.55

# Dataset_6
D6_ANCHOR_CLASSES  = ["A"]
D6_NEG_ONLY        = ["X"]
D6_XCLASS_NEG_PROB = 0.55
D6_A_OVERSAMPLE    = 3

# Image
IMAGE_SIZE = (256, 256)
NORM_MEAN  = [0.485, 0.456, 0.406]
NORM_STD   = [0.229, 0.224, 0.225]

# Model
BACKBONE   = "efficientnet_b0"
EMBED_DIM  = 256
PRETRAINED = True
USE_CBAM   = True

# Training
BATCH_SIZE     = 8
NUM_WORKERS    = 8
PIN_MEMORY     = True
VIRTUAL_EPOCH  = 20_000
TOTAL_EPOCHS   = 50
FREEZE_EPOCHS  = 5
WARMUP_EPOCHS  = 5
LR             = 3e-4
LR_BACKBONE    = 3e-5
WEIGHT_DECAY   = 1e-4
TRIPLET_MARGIN = 0.5

# Checkpointing
SAVE_EVERY_N = 5
SEED         = 42
