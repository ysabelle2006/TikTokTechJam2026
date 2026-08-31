import csv
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from tiktoktechjam2026.models.frequency_stream import FrequencyStream
from tiktoktechjam2026.models.fusion import ResidualFusionHead


EVAL_MANIFEST = Path(
    "data/eval_manifest.csv"
)

CHECKPOINT = Path(
    "checkpoints/v2_residual_fusion_best.pt"
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ============================================================
    # 1. Load clean shared-eval rows
    # ============================================================

    with open(
        EVAL_MANIFEST,
        newline="",
    ) as f:
        all_rows = list(
            csv.DictReader(f)
        )

    rows = [
        row
        for row in all_rows
        if row["condition"] == "clean"
    ]

    print(
        f"\nClean evaluation images: {len(rows)}"
    )

    # ============================================================
    # 2. Load checkpoint
    # ============================================================

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get("best_epoch", "?"),
    )

    # ============================================================
    # 3. Load trained SRM
    # ============================================================

    print(
        "Loading trained SRM..."
    )

    frequency = FrequencyStream(
        "srm"
    ).to(device)

    frequency.load_state_dict(
        checkpoint["frequency"]
    )

    frequency.eval()

    # ============================================================
    # 4. Load residual fusion head
    # ============================================================

    print(
        "Loading residual fusion head..."
    )

    fusion = ResidualFusionHead().to(
        device
    )

    fusion.load_state_dict(
        checkpoint["fusion"]
    )

    fusion.eval()

    print(
        f"Learned alpha: {fusion.alpha.item():.4f}"
    )

    # ============================================================
    # 5. Evaluate frequency correction ALONE
    # ============================================================

    labels = []
    correction_scores = []

    with torch.no_grad():

        for row in tqdm(
            rows,
            desc="Evaluating frequency correction",
        ):

            image = Image.open(
                row["transformed_path"]
            ).convert("RGB")

            image_tensor = (
                pil_to_tensor(image)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(device)
            )

            (
                frequency_embedding,
                residual_energy,
            ) = frequency(
                image_tensor
            )

            frequency_input = torch.cat(
                [
                    frequency_embedding,
                    residual_energy,
                ],
                dim=1,
            )

            correction = (
                fusion.frequency_correction(
                    frequency_input
                )
            )

            correction_scores.append(
                correction.item()
            )

            labels.append(
                int(row["label"])
            )

    # ============================================================
    # 6. AUC
    # ============================================================

    correction_auc = roc_auc_score(
        labels,
        correction_scores,
    )

    # If AUC < 0.5, also show how strong it would be
    # if its direction were reversed.
    reversed_auc = 1.0 - correction_auc

    print()
    print(
        "===================================="
    )
    print(
        "FREQUENCY CORRECTION CLEAN DIAGNOSTIC"
    )
    print(
        "===================================="
    )

    print(
        f"Images:                  {len(rows)}"
    )

    print(
        f"Correction-only AUC:     {correction_auc:.4f}"
    )

    print(
        f"Reverse-direction AUC:   {reversed_auc:.4f}"
    )

    print(
        f"Learned alpha:           {fusion.alpha.item():.4f}"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()