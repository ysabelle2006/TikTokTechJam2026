"""
Central configuration for the two-stream AIGC detector.

Keeping these values in one place makes it easy to swap the backbone,
change embedding sizes, or point at a different data directory without
hunting through every module later.

Every other module imports from here -- nothing downstream should
hardcode an image size, an embedding width, or a path.
"""

import os

IMAGE_SIZE = 224

# --------------------------------------------------------------------------
# Spatial stream
# --------------------------------------------------------------------------
# NOTE: the OpenAI CLIP weights were trained with the QuickGELU activation.
# open_clip's plain "ViT-B-32" config now defaults to nn.GELU and only warns
# about the mismatch, which silently degrades the embeddings. The
# "-quickgelu" variant loads the exact activation the weights expect.
SPATIAL_BACKBONE = "ViT-B-32-quickgelu"   # open_clip model name
SPATIAL_PRETRAINED = "openai"             # open_clip pretrained tag
SPATIAL_EMBED_DIM = 512
FREEZE_BACKBONE = True                    # flip to False (or partial) once we try fine-tuning (V4)

# --------------------------------------------------------------------------
# Frequency stream -- deliberately not committed to one extraction method yet.
# Run both as an ablation (see README roadmap) before picking a default.
# --------------------------------------------------------------------------
FREQUENCY_MODE = "srm"          # "srm" (high-pass residual) or "fft" (log-magnitude spectrum)
FREQUENCY_EMBED_DIM = 128
FREQUENCY_SRM_CHANNELS = 3      # number of SRM high-pass kernels stacked as CNN input (srm mode)

# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
FUSION_INPUT_DIM = SPATIAL_EMBED_DIM + FREQUENCY_EMBED_DIM + 1  # +1 = residual-energy scalar
FUSION_HIDDEN_DIMS = (256, 64)

# V0 spatial-only classification head (no frequency stream, no fusion).
SPATIAL_HEAD_HIDDEN_DIMS = (256, 64)

# --------------------------------------------------------------------------
# Training objective
# --------------------------------------------------------------------------
# V3: L = L_cls(y_hat, y) + L_cls(y_hat_t, y) + CONSISTENCY_LOSS_WEIGHT * L_consistency(y_hat, y_hat_t)
CONSISTENCY_LOSS_WEIGHT = 0.5   # lambda -- tune once L_cls and L_consistency are both being logged separately

# V0 / V1 head-training hyperparameters. The CLIP backbone is frozen, so we
# are only ever fitting a small head (V0) or a small CNN + MLP (V1).
BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 8         # epochs without val-AUC improvement before stopping
SEED = 42
DEVICE = os.environ.get("TTJ_DEVICE", "cpu")

# --------------------------------------------------------------------------
# Dataset (V0 / V1): SID_Set, binary real vs fully-synthetic.
# `tampered` (label 2) is dropped for these stages -- V0/V1 ask whether the
# streams separate genuine photos from fully generated images at all.
# --------------------------------------------------------------------------
SID_DIR = os.path.join("data", "sid_set", "train")
SID_CLASS_TO_LABEL = {
    "real": 0,             # genuine photograph
    "full_synthetic": 1,   # fully AI-generated
}
LABEL_NAMES = {0: "real", 1: "fake"}

SID_PER_CLASS_CAP = 8000               # max images per class pulled into the manifest (all of SID)
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)      # train / val / test, stratified, deterministic (SEED)
SPLIT_FILE = os.path.join("data", "sid_set_split.json")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
EMBEDDING_CACHE_DIR = os.path.join("cache", "spatial_embeddings")

CHECKPOINTS = {
    "v0": os.path.join(CHECKPOINT_DIR, "v0.pt"),
    "v1": os.path.join(CHECKPOINT_DIR, "v1.pt"),
}
RESULT_FILES = {
    "v0": os.path.join(RESULTS_DIR, "v0.json"),
    "v1": os.path.join(RESULTS_DIR, "v1.json"),
}

# --------------------------------------------------------------------------
# Robustness evaluation grid (the brief's transform x severity table).
#
# Each entry: (condition_key, transform_name, param). `clean` is the
# identity baseline. evaluate.py applies exactly ONE of these to the raw
# image, at native resolution, before either stream's preprocessing.
# These keys are the exact row labels of the "ROBUSTNESS SUMMARY" table
# and the keys persisted to results/v{0,1}.json.
# --------------------------------------------------------------------------
EVAL_CONDITIONS = [
    ("clean",           "identity", None),
    ("jpeg_q90",        "jpeg",     90),
    ("jpeg_q70",        "jpeg",     70),
    ("jpeg_q50",        "jpeg",     50),
    ("jpeg_q30",        "jpeg",     30),
    ("blur_0.5",        "blur",     0.5),
    ("blur_1.0",        "blur",     1.0),
    ("blur_2.0",        "blur",     2.0),
    ("resize_0.5",      "resize",   0.5),
    ("resize_0.25",     "resize",   0.25),
    ("noise_0.02",      "noise",    0.02),
    ("noise_0.05",      "noise",    0.05),
    ("noise_0.10",      "noise",    0.10),
    ("color_jitter_20", "color_jitter", 0.20),
    ("crop_0.8",        "crop",     0.8),
]

# Training-time augmentation (V2+) draws a random subset of these transform
# families -- never all of them at once (see augmentations.random_transform).
AUG_MAX_SIMULTANEOUS = 5
