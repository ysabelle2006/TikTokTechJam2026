from pathlib import Path

import torch
from torch import nn
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from tiktoktechjam2026.data.datasets import CIFAKEDataset
from tiktoktechjam2026.models.frequency_stream import FrequencyStream
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
    # 1. Load trained FFT model
    # ---------------------------------------------------------
    frequency = FrequencyStream("fft").to(device)
    classifier = nn.Linear(128, 1).to(device)

    checkpoint = torch.load(
        "checkpoints/v1_fft_best.pt",
        map_location=device,
    )

    frequency.load_state_dict(checkpoint["frequency"])
    classifier.load_state_dict(checkpoint["classifier"])

    frequency.eval()
    classifier.eval()

    # ---------------------------------------------------------
    # 2. Load SAME CIFAKE test set used before
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
    # 3. Evaluate one condition
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

            image_tensor = (
                pil_to_tensor(image)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                embedding, _ = frequency(image_tensor)

                logit = classifier(embedding).squeeze(1)
                probability = torch.sigmoid(logit)
                prediction = (probability >= 0.5).float()

            all_predictions.append(prediction.cpu())
            all_probabilities.append(probability.cpu())
            all_labels.append(
                torch.tensor([label], dtype=torch.float32)
            )

        predictions = torch.cat(all_predictions)
        probabilities = torch.cat(all_probabilities)
        labels = torch.cat(all_labels)

        return evaluate_predictions(
            predictions,
            probabilities,
            labels,
        )

    # ---------------------------------------------------------
    # 4. Clean + transformations
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
    # 5. Print summary
    # ---------------------------------------------------------
    clean_accuracy = results["clean"]["accuracy"]

    print("\nFFT ROBUSTNESS RESULTS")
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

    clean_auc = results["clean"]["auc"]

    transformed_aucs = [
        metrics["auc"]
        for name, metrics in results.items()
        if name != "clean"
    ]

    avg_robust_auc = sum(transformed_aucs) / len(transformed_aucs)

    final_score = 0.5 * clean_auc + 0.5 * avg_robust_auc

    print("\nclean AUC:      ", f"{clean_auc:.4f}")
    print("avg robust AUC: ", f"{avg_robust_auc:.4f}")
    print("final score:    ", f"{final_score:.4f}")

    # ---------------------------------------------------------
    # 6. Save results
    # ---------------------------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "v1_fft_robustness.txt"

    with open(output_path, "w") as f:
        f.write("V1 FFT Frequency-Only Ablation\n")
        f.write("Evaluation subset: 500 CIFAKE test images\n")
        f.write("250 REAL / 250 FAKE\n\n")

        f.write("FFT ROBUSTNESS RESULTS\n")
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
        f.write(f"clean AUC:       {clean_auc:.4f}\n")
        f.write(f"avg robust AUC:  {avg_robust_auc:.4f}\n")
        f.write(f"final score:     {final_score:.4f}\n")

    print("\nSaved results to:", output_path)


if __name__ == "__main__":
    main()