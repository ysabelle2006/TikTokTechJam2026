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
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from tiktoktechjam2026.data.datasets import CIFAKEDataset
from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.transforms.augmentations import (
    EVALUATION_TRANSFORMS,
    STACKED_TRANSFORMS,
)


def evaluate_predictions(predictions, probabilities, labels):
    accuracy = (predictions == labels).float().mean().item()

    true_positive = ((predictions == 1) & (labels == 1)).sum().item()
    true_negative = ((predictions == 0) & (labels == 0)).sum().item()
    false_positive = ((predictions == 1) & (labels == 0)).sum().item()
    false_negative = ((predictions == 0) & (labels == 1)).sum().item()

    precision = true_positive / max(
        true_positive + false_positive, 1
    )

    recall = true_positive / max(
        true_positive + false_negative, 1
    )

    f1 = 2 * precision * recall / max(
        precision + recall, 1e-8
    )

    auc = roc_auc_score(
        labels.numpy(),
        probabilities.numpy(),
    )

    return {
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # 1. Load trained V0 spatial classifier
    # ---------------------------------------------------------
    classifier = nn.Linear(512, 1).to(device)

    checkpoint = torch.load(
        "checkpoints/v0_spatial_best.pt",
        map_location=device,
    )

    classifier.load_state_dict(checkpoint)
    classifier.eval()

    # ---------------------------------------------------------
    # 2. Load frozen CLIP spatial encoder
    # ---------------------------------------------------------
    spatial = SpatialStream()

    spatial.model.eval()

    for param in spatial.model.parameters():
        param.requires_grad = False

    # ---------------------------------------------------------
    # 3. Load SAME CIFAKE test subset
    # ---------------------------------------------------------
    dataset = CIFAKEDataset(
        "data/CIFAKE",
        split="test",
    )

    real_indices = []
    fake_indices = []

    for i, (_, original_label) in enumerate(dataset.dataset.samples):
        class_name = dataset.dataset.classes[original_label].upper()

        if class_name == "REAL" and len(real_indices) < 250:
            real_indices.append(i)

        elif class_name == "FAKE" and len(fake_indices) < 250:
            fake_indices.append(i)

        if (
            len(real_indices) == 250
            and len(fake_indices) == 250
        ):
            break

    indices = real_indices + fake_indices

    print("Evaluation images:", len(indices))
    print("REAL:", len(real_indices))
    print("FAKE:", len(fake_indices))

    # ---------------------------------------------------------
    # 4. Evaluate one condition
    # ---------------------------------------------------------
    def run_condition(name, transform_fn=None):
        all_predictions = []
        all_probabilities = []
        all_labels = []

        for i in tqdm(
            indices,
            desc=f"Evaluating {name}",
        ):
            image, label = dataset[i]

            if transform_fn is not None:
                image = transform_fn(image)

            # CLIP preprocessing
            image_tensor = (
                spatial.preprocess(image)
                .unsqueeze(0)
            )

            with torch.no_grad():
                embedding = spatial.encode(
                    image_tensor
                )

                embedding = F.normalize(
                    embedding.float(),
                    dim=1,
                )

                embedding = embedding.to(device)

                logit = classifier(
                    embedding
                ).squeeze(1)

                probability = torch.sigmoid(
                    logit
                )

                prediction = (
                    probability >= 0.5
                ).float()

            all_predictions.append(
                prediction.cpu()
            )

            all_probabilities.append(
                probability.cpu()
            )

            all_labels.append(
                torch.tensor(
                    [label],
                    dtype=torch.float32,
                )
            )

        predictions = torch.cat(
            all_predictions
        )

        probabilities = torch.cat(
            all_probabilities
        )

        labels = torch.cat(
            all_labels
        )

        return evaluate_predictions(
            predictions,
            probabilities,
            labels,
        )

    # ---------------------------------------------------------
    # 5. Clean + transformations
    # ---------------------------------------------------------
    results = {}

    results["clean"] = run_condition(
        "clean",
        transform_fn=None,
    )

    for name, transform_fn in EVALUATION_TRANSFORMS.items():
        results[name] = run_condition(
            name,
            transform_fn,
        )

    for name, transform_fn in STACKED_TRANSFORMS.items():
        results[name] = run_condition(
            name,
            transform_fn,
        )

    # ---------------------------------------------------------
    # 6. Print summary
    # ---------------------------------------------------------
    clean_accuracy = results["clean"]["accuracy"]

    print("\nV0 SPATIAL ROBUSTNESS RESULTS")
    print("-" * 64)

    print(
        f"{'Condition':<22}"
        f"{'Accuracy':>10}"
        f"{'Drop':>10}"
        f"{'AUC':>10}"
    )

    print("-" * 64)

    for name, metrics in results.items():
        accuracy = metrics["accuracy"]
        drop = clean_accuracy - accuracy
        auc = metrics["auc"]

        print(
            f"{name:<22}"
            f"{accuracy:>10.3f}"
            f"{drop:>10.3f}"
            f"{auc:>10.4f}"
        )

    # ---------------------------------------------------------
    # 7. Summary scores
    # ---------------------------------------------------------
    clean_auc = results["clean"]["auc"]

    transformed_aucs = [
        metrics["auc"]
        for name, metrics in results.items()
        if name != "clean"
    ]

    avg_robust_auc = (
        sum(transformed_aucs)
        / len(transformed_aucs)
    )

    final_score = (
        0.5 * clean_auc
        + 0.5 * avg_robust_auc
    )

    print("\nclean AUC:      ", f"{clean_auc:.4f}")
    print("avg robust AUC: ", f"{avg_robust_auc:.4f}")
    print("final score:    ", f"{final_score:.4f}")

    # ---------------------------------------------------------
    # 8. Save results
    # ---------------------------------------------------------
    results_dir = Path("results")

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        results_dir
        / "v0_robustness.txt"
    )

    with open(output_path, "w") as f:
        f.write("V0 Spatial-Only Baseline\n")
        f.write(
            "Evaluation subset: "
            "500 CIFAKE test images\n"
        )
        f.write("250 REAL / 250 FAKE\n\n")

        f.write(
            "V0 SPATIAL ROBUSTNESS RESULTS\n"
        )

        f.write("-" * 64 + "\n")

        f.write(
            f"{'Condition':<22}"
            f"{'Accuracy':>10}"
            f"{'Drop':>10}"
            f"{'AUC':>10}\n"
        )

        f.write("-" * 64 + "\n")

        for name, metrics in results.items():
            accuracy = metrics["accuracy"]
            drop = clean_accuracy - accuracy
            auc = metrics["auc"]

            f.write(
                f"{name:<22}"
                f"{accuracy:>10.3f}"
                f"{drop:>10.3f}"
                f"{auc:>10.4f}\n"
            )

        f.write("\n")

        f.write(
            f"clean AUC:       "
            f"{clean_auc:.4f}\n"
        )

        f.write(
            f"avg robust AUC:  "
            f"{avg_robust_auc:.4f}\n"
        )

        f.write(
            f"final score:     "
            f"{final_score:.4f}\n"
        )

    print(
        "\nSaved results to:",
        output_path,
    )


if __name__ == "__main__":
    main()