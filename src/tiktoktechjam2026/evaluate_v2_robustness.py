import json
import random
from pathlib import Path

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
from tiktoktechjam2026.transforms.augmentations import (
    jpeg_compress,
    gaussian_blur,
    resize_roundtrip,
    gaussian_noise,
    color_jitter,
    center_crop,
)


# ==================================================
# Configuration
# ==================================================

SEED = 42

TEST_DIR = "data/cifake/test"

CHECKPOINT = "results/v3_transform_aware/best.pt"

OUTPUT_FILE = (
    "results/v3_transform_aware/"
    "cifake_robustness.json"
)

BATCH_SIZE = 64


# ==================================================
# Reproducibility
# ==================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==================================================
# Device
# ==================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# ==================================================
# Challenge conditions
# ==================================================

CONDITIONS = {

    "clean":
        None,

    "jpeg_q90":
        lambda img: jpeg_compress(
            img,
            90,
        ),

    "jpeg_q70":
        lambda img: jpeg_compress(
            img,
            70,
        ),

    "jpeg_q50":
        lambda img: jpeg_compress(
            img,
            50,
        ),

    "jpeg_q30":
        lambda img: jpeg_compress(
            img,
            30,
        ),

    "blur_0.5":
        lambda img: gaussian_blur(
            img,
            0.5,
        ),

    "blur_1.0":
        lambda img: gaussian_blur(
            img,
            1.0,
        ),

    "blur_2.0":
        lambda img: gaussian_blur(
            img,
            2.0,
        ),

    "resize_0.5":
        lambda img: resize_roundtrip(
            img,
            0.5,
        ),

    "resize_0.25":
        lambda img: resize_roundtrip(
            img,
            0.25,
        ),

    "noise_0.02":
        lambda img: gaussian_noise(
            img,
            0.02,
        ),

    "noise_0.05":
        lambda img: gaussian_noise(
            img,
            0.05,
        ),

    "noise_0.10":
        lambda img: gaussian_noise(
            img,
            0.10,
        ),

    "color_jitter":
        lambda img: color_jitter(
            img
        ),

    "center_crop_0.8":
        lambda img: center_crop(
            img,
            0.8,
        ),
}


# ==================================================
# Dataset
# ==================================================

class RobustnessDataset(Dataset):

    def __init__(
        self,
        root_dir,
        augmentation=None,
    ):

        self.base_dataset = (
            AIGCFolderDataset(
                root_dir=root_dir,
                transform=None,
                augmentation=None,
            )
        )

        self.augmentation = (
            augmentation
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
        ).convert("RGB")

        # Apply robustness transformation
        if self.augmentation is not None:

            image = self.augmentation(
                image
            )

        # Same transformed image goes into
        # BOTH spatial and frequency branches
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
        )


# ==================================================
# Evaluate one condition
# ==================================================

def evaluate_condition(
    condition_name,
    augmentation,
    spatial_stream,
    frequency_stream,
    fusion_head,
):

    print()
    print("=" * 60)
    print(
        f"EVALUATING: {condition_name}"
    )
    print("=" * 60)

    # Reset seeds so random transforms
    # are reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = RobustnessDataset(
        TEST_DIR,
        augmentation=augmentation,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_labels = []
    all_probabilities = []

    frequency_stream.eval()
    fusion_head.eval()

    with torch.no_grad():

        for batch_index, (
            spatial_images,
            frequency_images,
            labels,
        ) in enumerate(loader):

            # ==========================================
            # Spatial branch
            # ==========================================

            spatial_embeddings = (
                spatial_stream.encode(
                    spatial_images
                )
            )

            # ==========================================
            # Frequency branch
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

            if (
                (batch_index + 1)
                % 100
                == 0
            ):

                print(
                    f"Processed "
                    f"{batch_index + 1}"
                    f"/{len(loader)} batches"
                )

    # ==================================================
    # Metrics
    # ==================================================

    labels = np.array(
        all_labels
    )

    probabilities = np.array(
        all_probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = (
        predictions == labels
    ).mean()

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    # REAL = 0
    # FAKE = 1

    false_positives = (
        (predictions == 1)
        & (labels == 0)
    ).sum()

    true_negatives = (
        (predictions == 0)
        & (labels == 0)
    ).sum()

    false_negatives = (
        (predictions == 0)
        & (labels == 1)
    ).sum()

    true_positives = (
        (predictions == 1)
        & (labels == 1)
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

    print()
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

    return {

        "accuracy":
            float(accuracy),

        "auc":
            float(auc),

        "fpr":
            float(fpr),

        "fnr":
            float(fnr),
    }


# ==================================================
# Main
# ==================================================

def main():

    print()
    print("=" * 60)
    print(
        "LOADING V2 MULTI-DOMAIN MODEL"
    )
    print("=" * 60)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    frequency_mode = (
        checkpoint[
            "frequency_mode"
        ]
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )

    print(
        "Frequency mode:",
        frequency_mode,
    )

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
        ).to(device)
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
    # Run all robustness conditions
    # ==================================================

    results = {}

    for (
        condition_name,
        augmentation,
    ) in CONDITIONS.items():

        results[
            condition_name
        ] = evaluate_condition(
            condition_name,
            augmentation,
            spatial_stream,
            frequency_stream,
            fusion_head,
        )

    # ==================================================
    # Calculate robustness summary
    # ==================================================

    clean_accuracy = (
        results[
            "clean"
        ][
            "accuracy"
        ]
    )

    transformed_accuracies = []

    for (
        condition_name,
        metrics,
    ) in results.items():

        if condition_name != "clean":

            transformed_accuracies.append(
                metrics[
                    "accuracy"
                ]
            )

            metrics[
                "accuracy_drop_from_clean"
            ] = float(
                clean_accuracy
                - metrics[
                    "accuracy"
                ]
            )

    average_transformed_accuracy = (
        np.mean(
            transformed_accuracies
        )
    )

    worst_condition = min(
        (
            name
            for name
            in results
            if name != "clean"
        ),
        key=lambda name:
            results[name][
                "accuracy"
            ],
    )

    worst_accuracy = (
        results[
            worst_condition
        ][
            "accuracy"
        ]
    )

    summary = {

        "clean_accuracy":
            float(
                clean_accuracy
            ),

        "average_transformed_accuracy":
            float(
                average_transformed_accuracy
            ),

        "worst_condition":
            worst_condition,

        "worst_accuracy":
            float(
                worst_accuracy
            ),
    }

    # ==================================================
    # Print final table
    # ==================================================

    print()
    print()
    print("=" * 75)
    print(
        "FINAL CIFAKE ROBUSTNESS SUMMARY"
    )
    print("=" * 75)

    print(
        f"{'Condition':<20}"
        f"{'Accuracy':>12}"
        f"{'AUC':>12}"
        f"{'FPR':>12}"
        f"{'FNR':>12}"
    )

    print("-" * 75)

    for (
        condition_name,
        metrics,
    ) in results.items():

        print(
            f"{condition_name:<20}"
            f"{metrics['accuracy'] * 100:>11.2f}%"
            f"{metrics['auc']:>12.4f}"
            f"{metrics['fpr'] * 100:>11.2f}%"
            f"{metrics['fnr'] * 100:>11.2f}%"
        )

    print()
    print(
        f"Clean accuracy: "
        f"{clean_accuracy * 100:.2f}%"
    )

    print(
        "Average transformed accuracy: "
        f"{average_transformed_accuracy * 100:.2f}%"
    )

    print(
        f"Worst condition: "
        f"{worst_condition}"
    )

    print(
        f"Worst accuracy: "
        f"{worst_accuracy * 100:.2f}%"
    )

    # ==================================================
    # Save JSON
    # ==================================================

    output = {

        "checkpoint":
            CHECKPOINT,

        "test_dataset":
            TEST_DIR,

        "results":
            results,

        "summary":
            summary,
    }

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
            output,
            f,
            indent=4,
        )

    print()
    print(
        "Saved results to:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()