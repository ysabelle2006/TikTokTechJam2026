"""
Robustness evaluation harness for V1.

Evaluates the two-stream detector:
    spatial CLIP embedding
    + frequency stream (SRM-style or FFT)
    + fusion head

Runs on the clean CIFAKE test set and on every robustness condition
from the challenge brief.

Outputs:
    results/v1_srm_robustness.json
or:
    results/v1_fft_robustness.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score

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
    center_crop,
)


# ==================================================
# Configuration
# ==================================================

TEST_DIR = "data/cifake/test"

# Change only this when switching SRM <-> FFT
CHECKPOINT = "results/v1_fft/best.pt"
OUTPUT_FILE = "results/v1_fft_robustness.json"


# ==================================================
# Deterministic color jitter
# ==================================================

def fixed_color_jitter(image):
    """
    Deterministic +20% brightness, contrast,
    and saturation.
    """
    from PIL import ImageEnhance

    image = ImageEnhance.Brightness(
        image
    ).enhance(1.2)

    image = ImageEnhance.Contrast(
        image
    ).enhance(1.2)

    image = ImageEnhance.Color(
        image
    ).enhance(1.2)

    return image


# ==================================================
# Robustness conditions
# ==================================================

CONDITIONS = {
    "clean": None,

    "jpeg_q90":
        lambda img: jpeg_compress(img, 90),

    "jpeg_q70":
        lambda img: jpeg_compress(img, 70),

    "jpeg_q50":
        lambda img: jpeg_compress(img, 50),

    "jpeg_q30":
        lambda img: jpeg_compress(img, 30),

    "blur_0.5":
        lambda img: gaussian_blur(img, 0.5),

    "blur_1.0":
        lambda img: gaussian_blur(img, 1.0),

    "blur_2.0":
        lambda img: gaussian_blur(img, 2.0),

    "resize_0.5":
        lambda img: resize_roundtrip(img, 0.5),

    "resize_0.25":
        lambda img: resize_roundtrip(img, 0.25),

    "noise_0.02":
        lambda img: gaussian_noise(img, 0.02),

    "noise_0.05":
        lambda img: gaussian_noise(img, 0.05),

    "noise_0.10":
        lambda img: gaussian_noise(img, 0.10),

    "color_jitter_20":
        fixed_color_jitter,

    "crop_0.8":
        lambda img: center_crop(img, 0.8),
}


# ==================================================
# Dataset returning BOTH views of same image
# ==================================================

class V1EvaluationDataset(Dataset):

    def __init__(
        self,
        root_dir,
        augmentation=None,
    ):
        self.base_dataset = AIGCFolderDataset(
            root_dir=root_dir,
            augmentation=None,
            transform=None,
        )

        self.augmentation = augmentation

        self.frequency_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):

        image_path, label = (
            self.base_dataset.samples[index]
        )

        from PIL import Image

        image = Image.open(
            image_path
        ).convert("RGB")

        # Apply SAME robustness transform first
        if self.augmentation is not None:
            image = self.augmentation(image)

        # Spatial branch
        spatial_image = prepare_spatial_input(
            image
        )

        # Frequency branch
        frequency_image = (
            self.frequency_transform(image)
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
    device,
):

    print()
    print(f"Evaluating: {condition_name}")

    # Make stochastic noise reproducible
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = V1EvaluationDataset(
        root_dir=TEST_DIR,
        augmentation=augmentation,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for batch_index, (
        spatial_images,
        frequency_images,
        labels,
    ) in enumerate(loader):

        # ----------------------------------
        # Spatial CLIP branch
        # ----------------------------------

        spatial_embeddings = (
            spatial_stream.encode(
                spatial_images
            )
        )

        # ----------------------------------
        # Frequency branch
        # ----------------------------------

        frequency_images = (
            frequency_images.to(device)
        )

        with torch.no_grad():

            frequency_embeddings, residual_energy = (
                frequency_stream(
                    frequency_images
                )
            )

            # ----------------------------------
            # Fusion
            # ----------------------------------

            logits = fusion_head(
                spatial_embeddings,
                frequency_embeddings,
                residual_energy,
            ).squeeze(1)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).long()

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        if (
            (batch_index + 1) % 50 == 0
            or batch_index == 0
        ):
            print(
                f"  Processed batch "
                f"{batch_index + 1} / "
                f"{len(loader)}"
            )

    # ==================================================
    # Metrics
    # ==================================================

    labels = np.array(
        all_labels
    )

    predictions = np.array(
        all_predictions
    )

    probabilities = np.array(
        all_probabilities
    )

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
        / (
            false_positives
            + true_negatives
        )
    )

    fnr = (
        false_negatives
        / (
            false_negatives
            + true_positives
        )
    )

    print(
        f"  Accuracy: {accuracy:.4f} | "
        f"AUC: {auc:.4f} | "
        f"FPR: {fpr:.4f} | "
        f"FNR: {fnr:.4f}"
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "auc": float(
            auc
        ),
        "fpr": float(
            fpr
        ),
        "fnr": float(
            fnr
        ),
    }


# ==================================================
# Main
# ==================================================

def main():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # ----------------------------------
    # Load checkpoint
    # ----------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    frequency_mode = checkpoint[
        "frequency_mode"
    ]

    print(
        "Frequency mode:",
        frequency_mode
    )

    print(
        "Checkpoint validation accuracy:",
        checkpoint["val_accuracy"]
    )

    # ----------------------------------
    # Frozen CLIP spatial stream
    # ----------------------------------

    spatial_stream = SpatialStream(
        freeze=True
    )

    # ----------------------------------
    # Frequency stream
    # ----------------------------------

    frequency_stream = FrequencyStream(
        mode=frequency_mode
    ).to(device)

    frequency_stream.load_state_dict(
        checkpoint[
            "frequency_stream"
        ]
    )

    frequency_stream.eval()

    # ----------------------------------
    # Fusion head
    # ----------------------------------

    fusion_head = FusionHead().to(
        device
    )

    fusion_head.load_state_dict(
        checkpoint[
            "fusion_head"
        ]
    )

    fusion_head.eval()

    # ----------------------------------
    # Evaluate every condition
    # ----------------------------------

    results = {}

    for (
        condition_name,
        augmentation,
    ) in CONDITIONS.items():

        metrics = evaluate_condition(
            condition_name,
            augmentation,
            spatial_stream,
            frequency_stream,
            fusion_head,
            device,
        )

        results[
            condition_name
        ] = metrics

    # ==================================================
    # Accuracy drop
    # ==================================================

    clean_accuracy = (
        results["clean"]["accuracy"]
    )

    for condition_name in results:

        accuracy = results[
            condition_name
        ]["accuracy"]

        results[
            condition_name
        ]["accuracy_drop"] = float(
            clean_accuracy
            - accuracy
        )

    # ==================================================
    # Save JSON
    # ==================================================

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

    # ==================================================
    # Summary
    # ==================================================

    print()
    print("=" * 72)

    print(
        f"V1-{frequency_mode.upper()} "
        f"ROBUSTNESS SUMMARY"
    )

    print("=" * 72)

    print(
        f"{'Condition':<20}"
        f"{'Accuracy':>12}"
        f"{'Drop':>12}"
        f"{'AUC':>12}"
    )

    print("-" * 72)

    for (
        condition_name,
        metrics,
    ) in results.items():

        print(
            f"{condition_name:<20}"
            f"{metrics['accuracy'] * 100:>11.2f}%"
            f"{metrics['accuracy_drop'] * 100:>11.2f}%"
            f"{metrics['auc']:>12.4f}"
        )

    print()

    print(
        "Saved results to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()