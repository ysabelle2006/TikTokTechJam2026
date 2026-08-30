import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tiktoktechjam2026.data.datasets import AIGCFolderDataset
from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.models.frequency_stream import FrequencyStream
from tiktoktechjam2026.models.fusion import FusionHead
from tiktoktechjam2026.transforms.preprocessing import prepare_spatial_input


# ==================================================
# Configuration
# ==================================================

SID_DIR = "data/sid_subset"

CHECKPOINT = (
    "results/v3_transform_aware/best.pt"
)

OUTPUT_FILE = (
    "results/v3_transform_aware/"
    "sid_test.json"
)

PLOT_FILE = (
    "results/v3_transform_aware/"
    "sid_probability_distribution.png"
)

BATCH_SIZE = 64


# ==================================================
# Device
# ==================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(
    "Using device:",
    device,
)


# ==================================================
# Dataset
# ==================================================

class SIDDataset(Dataset):

    def __init__(
        self,
        root_dir,
    ):

        self.base_dataset = (
            AIGCFolderDataset(
                root_dir=root_dir,
                transform=None,
                augmentation=None,
            )
        )

        self.frequency_transform = (
            transforms.Compose([
                transforms.Resize(
                    (224, 224)
                ),
                transforms.ToTensor(),
            ])
        )

    def __len__(self):

        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        index,
    ):

        image_path, label = (
            self.base_dataset.samples[
                index
            ]
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        spatial_image = (
            prepare_spatial_input(
                image
            )
        )

        frequency_image = (
            self.frequency_transform(
                image
            )
        )

        return (
            spatial_image,
            frequency_image,
            label,
            str(image_path),
        )


# ==================================================
# Load model
# ==================================================

print()
print("=" * 60)
print("LOADING V3 MODEL")
print("=" * 60)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
)

frequency_mode = checkpoint[
    "frequency_mode"
]

print(
    "Checkpoint:",
    CHECKPOINT,
)

print(
    "Frequency mode:",
    frequency_mode,
)

if "val_accuracy" in checkpoint:

    print(
        "Validation accuracy:",
        checkpoint[
            "val_accuracy"
        ],
    )


# ==================================================
# Spatial stream
# ==================================================

spatial_stream = (
    SpatialStream(
        freeze=True
    )
)


# ==================================================
# Frequency stream
# ==================================================

frequency_stream = (
    FrequencyStream(
        mode=frequency_mode
    ).to(
        device
    )
)

frequency_stream.load_state_dict(
    checkpoint[
        "frequency_stream"
    ]
)

frequency_stream.eval()


# ==================================================
# Fusion head
# ==================================================

fusion_head = (
    FusionHead().to(
        device
    )
)

fusion_head.load_state_dict(
    checkpoint[
        "fusion_head"
    ]
)

fusion_head.eval()


# ==================================================
# Dataset + loader
# ==================================================

dataset = SIDDataset(
    SID_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print()
print(
    "SID evaluation images:",
    len(dataset),
)


# ==================================================
# Inference
# ==================================================

all_labels = []
all_probabilities = []
all_paths = []

with torch.no_grad():

    for batch_index, (
        spatial_images,
        frequency_images,
        labels,
        image_paths,
    ) in enumerate(
        loader
    ):

        # ==========================================
        # Spatial
        # ==========================================

        spatial_embeddings = (
            spatial_stream.encode(
                spatial_images
            )
        )

        # ==========================================
        # Frequency
        # ==========================================

        frequency_images = (
            frequency_images.to(
                device
            )
        )

        (
            frequency_embeddings,
            residual_energy,
        ) = frequency_stream(
            frequency_images
        )

        # ==========================================
        # Fusion
        # ==========================================

        logits = fusion_head(
            spatial_embeddings,
            frequency_embeddings,
            residual_energy,
        ).squeeze(1)

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        all_labels.extend(
            labels.numpy()
        )

        all_probabilities.extend(
            probabilities
            .cpu()
            .numpy()
        )

        all_paths.extend(
            image_paths
        )

        if (
            (batch_index + 1)
            % 10
            == 0
        ):

            print(
                f"Processed "
                f"{batch_index + 1}"
                f"/{len(loader)} batches"
            )


# ==================================================
# Convert to arrays
# ==================================================

labels = np.array(
    all_labels
)

probabilities = np.array(
    all_probabilities
)


# ==================================================
# Default threshold metrics
# ==================================================

DEFAULT_THRESHOLD = 0.5

predictions = (
    probabilities
    >= DEFAULT_THRESHOLD
).astype(
    int
)

accuracy = (
    predictions
    == labels
).mean()

auc = roc_auc_score(
    labels,
    probabilities,
)


# ==================================================
# Confusion components
#
# REAL = 0
# FAKE = 1
# ==================================================

false_positives = (
    (
        predictions == 1
    )
    &
    (
        labels == 0
    )
).sum()

true_negatives = (
    (
        predictions == 0
    )
    &
    (
        labels == 0
    )
).sum()

false_negatives = (
    (
        predictions == 0
    )
    &
    (
        labels == 1
    )
).sum()

true_positives = (
    (
        predictions == 1
    )
    &
    (
        labels == 1
    )
).sum()

fpr = (
    false_positives
    /
    (
        false_positives
        + true_negatives
    )
)

fnr = (
    false_negatives
    /
    (
        false_negatives
        + true_positives
    )
)


# ==================================================
# Print generalisation results
# ==================================================

print()
print("=" * 60)
print("SID_SET GENERALISATION RESULTS")
print("=" * 60)

print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"AUC:      "
    f"{auc:.4f}"
)

print(
    f"FPR:      "
    f"{fpr * 100:.2f}%"
)

print(
    f"FNR:      "
    f"{fnr * 100:.2f}%"
)


# ==================================================
# Threshold sweep
#
# IMPORTANT:
# This is analysis only.
# Do NOT report this as unbiased test accuracy,
# since we are selecting the threshold using
# the same evaluation set.
# ==================================================

thresholds = np.linspace(
    0.0,
    1.0,
    1001,
)

best_threshold = (
    DEFAULT_THRESHOLD
)

best_accuracy = (
    accuracy
)

for threshold in thresholds:

    threshold_predictions = (
        probabilities
        >= threshold
    ).astype(
        int
    )

    threshold_accuracy = (
        threshold_predictions
        == labels
    ).mean()

    if (
        threshold_accuracy
        > best_accuracy
    ):

        best_accuracy = (
            threshold_accuracy
        )

        best_threshold = (
            threshold
        )


print()
print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

print(
    f"Default threshold: "
    f"{DEFAULT_THRESHOLD:.3f}"
)

print(
    f"Best threshold:    "
    f"{best_threshold:.3f}"
)

print(
    f"Accuracy @ 0.5:    "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Best accuracy:     "
    f"{best_accuracy * 100:.2f}%"
)


# ==================================================
# Probability distribution
# ==================================================

real_probabilities = (
    probabilities[
        labels == 0
    ]
)

fake_probabilities = (
    probabilities[
        labels == 1
    ]
)

print()
print("=" * 60)
print("PROBABILITY DISTRIBUTION")
print("=" * 60)

print(
    f"REAL mean probability: "
    f"{real_probabilities.mean():.6f}"
)

print(
    f"REAL median:           "
    f"{np.median(real_probabilities):.6f}"
)

print(
    f"REAL min/max:          "
    f"{real_probabilities.min():.6f}"
    f" / "
    f"{real_probabilities.max():.6f}"
)

print()

print(
    f"FAKE mean probability: "
    f"{fake_probabilities.mean():.6f}"
)

print(
    f"FAKE median:           "
    f"{np.median(fake_probabilities):.6f}"
)

print(
    f"FAKE min/max:          "
    f"{fake_probabilities.min():.6f}"
    f" / "
    f"{fake_probabilities.max():.6f}"
)


# ==================================================
# Save histogram
# ==================================================

plot_path = Path(
    PLOT_FILE
)

plot_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.figure(
    figsize=(
        9,
        6,
    )
)

plt.hist(
    real_probabilities,
    bins=50,
    alpha=0.6,
    label="REAL",
)

plt.hist(
    fake_probabilities,
    bins=50,
    alpha=0.6,
    label="FAKE",
)

plt.xlabel(
    "Predicted AIGC probability"
)

plt.ylabel(
    "Number of images"
)

plt.title(
    "V3 SID_Set Probability Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    plot_path
)

plt.close()

print()
print(
    "Saved probability plot to:",
    PLOT_FILE,
)


# ==================================================
# Save results JSON
# ==================================================

results = {

    "checkpoint":
        CHECKPOINT,

    "dataset":
        SID_DIR,

    "num_images":
        int(
            len(dataset)
        ),

    "default_threshold":
        float(
            DEFAULT_THRESHOLD
        ),

    "accuracy":
        float(
            accuracy
        ),

    "auc":
        float(
            auc
        ),

    "fpr":
        float(
            fpr
        ),

    "fnr":
        float(
            fnr
        ),

    "confusion_matrix": {

        "true_positive":
            int(
                true_positives
            ),

        "true_negative":
            int(
                true_negatives
            ),

        "false_positive":
            int(
                false_positives
            ),

        "false_negative":
            int(
                false_negatives
            ),
    },

    "threshold_analysis": {

        "best_threshold":
            float(
                best_threshold
            ),

        "accuracy_at_default_threshold":
            float(
                accuracy
            ),

        "best_accuracy":
            float(
                best_accuracy
            ),
    },

    "probability_distribution": {

        "real": {

            "mean":
                float(
                    real_probabilities.mean()
                ),

            "median":
                float(
                    np.median(
                        real_probabilities
                    )
                ),

            "min":
                float(
                    real_probabilities.min()
                ),

            "max":
                float(
                    real_probabilities.max()
                ),
        },

        "fake": {

            "mean":
                float(
                    fake_probabilities.mean()
                ),

            "median":
                float(
                    np.median(
                        fake_probabilities
                    )
                ),

            "min":
                float(
                    fake_probabilities.min()
                ),

            "max":
                float(
                    fake_probabilities.max()
                ),
        },
    },

    "predictions": [],
}


# ==================================================
# Save per-image predictions
# ==================================================

for (
    image_path,
    label,
    probability,
) in zip(
    all_paths,
    labels,
    probabilities,
):

    results[
        "predictions"
    ].append(
        {
            "image_path":
                image_path,

            "label":
                int(label),

            "aigc_probability":
                float(
                    probability
                ),

            "prediction":
                int(
                    probability
                    >= DEFAULT_THRESHOLD
                ),
        }
    )


output_path = Path(
    OUTPUT_FILE
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=4,
    )

print()
print(
    "Saved results to:",
    OUTPUT_FILE,
)