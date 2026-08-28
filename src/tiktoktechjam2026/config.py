"""
Central configuration for the two-stream AIGC detector.

Keeping these values in one place makes it easy to swap the backbone,
change embedding sizes, or point at a different data directory without
hunting through every module later.
"""

IMAGE_SIZE = 224

# Spatial stream
SPATIAL_BACKBONE = "ViT-B-32"   # open_clip model name
SPATIAL_PRETRAINED = "openai"   # open_clip pretrained tag
SPATIAL_EMBED_DIM = 512
FREEZE_BACKBONE = True          # flip to False (or partial) once we try fine-tuning (V4)

# Frequency stream -- deliberately not committed to one extraction method yet.
# Run both as an ablation (see README roadmap) before picking a default.
FREQUENCY_MODE = "srm"          # "srm" (high-pass residual) or "fft" (log-magnitude spectrum)
FREQUENCY_EMBED_DIM = 128

# Fusion
FUSION_INPUT_DIM = SPATIAL_EMBED_DIM + FREQUENCY_EMBED_DIM + 1  # +1 = residual-energy scalar
FUSION_HIDDEN_DIMS = (256, 64)

# Training objective (V3): L = L_cls(y_hat, y) + L_cls(y_hat_t, y) + CONSISTENCY_LOSS_WEIGHT * L_consistency(y_hat, y_hat_t)
CONSISTENCY_LOSS_WEIGHT = 0.5    # lambda -- tune once L_cls and L_consistency are both being logged separately

# Paths
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
EMBEDDING_CACHE_DIR = "cache/spatial_embeddings"
