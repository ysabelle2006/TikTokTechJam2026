"""
Robustness evaluation harness.

Runs whichever detector variant (V0-V4) on the clean test set and on
each transform x severity combination from the brief, logging
accuracy / AUC / false-positive-rate / false-negative-rate. Doubles as
both the "Robustness Evaluation Summary" deliverable and the ablation
table that makes the incremental-improvement story legible:

    Detector             | Clean | JPEG | Blur | Resize | Avg robust drop
    ----------------------+-------+------+------+--------+-----------------
    Spatial only (V0)     |       |      |      |        |
    Frequency only        |       |      |      |        |
    Spatial + frequency   |       |      |      |        |
    + augmentation (V2)   |       |      |      |        |
    + consistency (V3)    |       |      |      |        |

Write results for each stage to results/<stage>.json rather than
overwriting a single file -- see results/README.md.

TODO: build once training + a first checkpoint (V0) exist.
"""


import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from tiktoktechjam2026.data.datasets import AIGCFolderDataset
from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.models.v0_classifier import V0Classifier
from tiktoktechjam2026.transforms.preprocessing import prepare_spatial_input
from tiktoktechjam2026.transforms.augmentations import (
    jpeg_compress,
    gaussian_blur,
    resize_roundtrip,
    gaussian_noise,
    center_crop,
)


TEST_DIR = "data/cifake/test"
CHECKPOINT = "results/v0/v0_classifier_best.pt"
OUTPUT_FILE = "results/v0_robustness.json"


def fixed_color_jitter(image):
    """
    Deterministic +20% brightness, contrast, and saturation.
    Evaluation should be reproducible rather than randomly jittered.
    """
    from PIL import ImageEnhance

    image = ImageEnhance.Brightness(image).enhance(1.2)
    image = ImageEnhance.Contrast(image).enhance(1.2)
    image = ImageEnhance.Color(image).enhance(1.2)

    return image


# Every robustness condition from the challenge brief
CONDITIONS = {
    "clean": None,

    "jpeg_q90": lambda img: jpeg_compress(img, 90),
    "jpeg_q70": lambda img: jpeg_compress(img, 70),
    "jpeg_q50": lambda img: jpeg_compress(img, 50),
    "jpeg_q30": lambda img: jpeg_compress(img, 30),

    "blur_0.5": lambda img: gaussian_blur(img, 0.5),
    "blur_1.0": lambda img: gaussian_blur(img, 1.0),
    "blur_2.0": lambda img: gaussian_blur(img, 2.0),

    "resize_0.5": lambda img: resize_roundtrip(img, 0.5),
    "resize_0.25": lambda img: resize_roundtrip(img, 0.25),

    "noise_0.02": lambda img: gaussian_noise(img, 0.02),
    "noise_0.05": lambda img: gaussian_noise(img, 0.05),
    "noise_0.10": lambda img: gaussian_noise(img, 0.10),

    "color_jitter_20": fixed_color_jitter,

    "crop_0.8": lambda img: center_crop(img, 0.8),
}


def evaluate_condition(
    condition_name,
    augmentation,
    spatial_stream,
    classifier,
    device,
):
    print()
    print(f"Evaluating: {condition_name}")

    dataset = AIGCFolderDataset(
        root_dir=TEST_DIR,
        augmentation=augmentation,
        transform=prepare_spatial_input,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for batch_index, (images, labels) in enumerate(loader):
        # CLIP embedding
        embeddings = spatial_stream.encode(images)

        # Classifier prediction
        with torch.no_grad():
            logits = classifier(embeddings)

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

        if (batch_index + 1) % 50 == 0 or batch_index == 0:
            print(
                f"  Processed batch "
                f"{batch_index + 1} / {len(loader)}"
            )

    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)

    accuracy = (predictions == labels).mean()

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    # REAL = 0
    # FAKE = 1
    false_positives = (
        (predictions == 1) & (labels == 0)
    ).sum()

    true_negatives = (
        (predictions == 0) & (labels == 0)
    ).sum()

    false_negatives = (
        (predictions == 0) & (labels == 1)
    ).sum()

    true_positives = (
        (predictions == 1) & (labels == 1)
    ).sum()

    fpr = false_positives / (
        false_positives + true_negatives
    )

    fnr = false_negatives / (
        false_negatives + true_positives
    )

    print(
        f"  Accuracy: {accuracy:.4f} | "
        f"AUC: {auc:.4f} | "
        f"FPR: {fpr:.4f} | "
        f"FNR: {fnr:.4f}"
    )

    return {
        "accuracy": float(accuracy),
        "auc": float(auc),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def main():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # Frozen CLIP backbone
    spatial_stream = SpatialStream(
        freeze=True
    )

    # V0 classifier
    classifier = V0Classifier().to(device)

    state_dict = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    classifier.load_state_dict(state_dict)
    classifier.eval()

    results = {}

    # -------------------------
    # Evaluate every condition
    # -------------------------

    for condition_name, augmentation in CONDITIONS.items():

        metrics = evaluate_condition(
            condition_name,
            augmentation,
            spatial_stream,
            classifier,
            device,
        )

        results[condition_name] = metrics

    # -------------------------
    # Calculate robustness drop
    # -------------------------

    clean_accuracy = results["clean"]["accuracy"]

    for condition_name in results:

        accuracy = results[
            condition_name
        ]["accuracy"]

        results[
            condition_name
        ]["accuracy_drop"] = float(
            clean_accuracy - accuracy
        )

    # -------------------------
    # Save JSON
    # -------------------------

    output_path = Path(OUTPUT_FILE)

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

    # -------------------------
    # Summary
    # -------------------------

    print()
    print("=" * 72)
    print("V0 ROBUSTNESS SUMMARY")
    print("=" * 72)

    print(
        f"{'Condition':<20}"
        f"{'Accuracy':>12}"
        f"{'Drop':>12}"
        f"{'AUC':>12}"
    )

    print("-" * 72)

    for condition_name, metrics in results.items():

        print(
            f"{condition_name:<20}"
            f"{metrics['accuracy'] * 100:>11.2f}%"
            f"{metrics['accuracy_drop'] * 100:>11.2f}%"
            f"{metrics['auc']:>12.4f}"
        )

    print()
    print("Saved results to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()