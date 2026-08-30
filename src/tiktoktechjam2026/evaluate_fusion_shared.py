import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.models.frequency_stream import FrequencyStream
from tiktoktechjam2026.models.fusion import FusionHead


# ============================================================
# Paths
# ============================================================

EVAL_MANIFEST = Path("data/eval_manifest.csv")

SHARED_CHECKPOINT = Path(
    "checkpoints/v1_fusion_sharedtrain_best.pt"
)

RESULT_PATH = Path(
    "results/v1_fusion_sharedtrain_robustness.txt"
)


# Ysabelle's held-out-generator evaluation:
# COCO = real
# WildFake DALL-E = fake from generator unseen during training
UNSEEN_GENERATOR_SOURCES = {
    "coco_val2017",
    "wildfake_dalle",
}


# ============================================================
# Load eval manifest
# ============================================================

def load_eval_manifest():
    if not EVAL_MANIFEST.is_file():
        raise FileNotFoundError(
            f"{EVAL_MANIFEST} not found."
        )

    with open(EVAL_MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))

    return rows


# ============================================================
# Main evaluation
# ============================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Load shared evaluation manifest
    # --------------------------------------------------------

    rows = load_eval_manifest()

    print(f"\nLoaded {len(rows)} evaluation rows.")

    condition_counts = Counter(
        row["condition"] for row in rows
    )

    print("\nCondition counts:")

    for condition, count in sorted(condition_counts.items()):
        print(f"  {condition:<20} {count}")

    # --------------------------------------------------------
    # Check transformed files exist
    # --------------------------------------------------------

    missing = [
        row["transformed_path"]
        for row in rows
        if not Path(row["transformed_path"]).is_file()
    ]

    if missing:
        print("\nExample missing path:")
        print(missing[0])

        raise FileNotFoundError(
            f"{len(missing)} transformed evaluation images "
            f"could not be found."
        )

    print("\nAll transformed evaluation images found.")

    # --------------------------------------------------------
    # Load Spatial Stream
    # --------------------------------------------------------

    print("\nLoading spatial stream...")

    spatial = SpatialStream()

    spatial.model.eval()

    for parameter in spatial.model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # Load SRM Frequency Stream
    # --------------------------------------------------------

    # --------------------------------------------------------
# Load jointly-trained SRM + Fusion checkpoint
# --------------------------------------------------------

    print("Loading jointly-trained SRM + fusion...")

    checkpoint = torch.load(
        SHARED_CHECKPOINT,
        map_location=device,
    )

# SRM frequency stream
    frequency = FrequencyStream("srm").to(device)

    frequency.load_state_dict(
        checkpoint["frequency"]
    )

    frequency.eval()

# Fusion head
    fusion = FusionHead().to(device)

    fusion.load_state_dict(
        checkpoint["fusion"]
    )

    fusion.eval()

    print(
        "Loaded checkpoint from epoch:",
        checkpoint.get("best_epoch", "?"),
    )

    print(
        "Checkpoint validation loss:",
        checkpoint.get("best_val_loss", "?"),
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    labels = []
    probabilities = []
    conditions = []
    sources = []

    print(
        f"\nEvaluating {len(rows)} transformed images..."
    )

    with torch.no_grad():

        for row in tqdm(rows):

            image_path = Path(
                row["transformed_path"]
            )

            image = Image.open(
                image_path
            ).convert("RGB")

            # ------------------------------
            # Spatial / CLIP branch
            # ------------------------------

            spatial_tensor = (
                spatial.preprocess(image)
                .unsqueeze(0)
            )

            spatial_embedding = spatial.encode(
                spatial_tensor
            )

            spatial_embedding = F.normalize(
                spatial_embedding.float(),
                dim=1,
            )

            spatial_embedding = spatial_embedding.to(
                device
            )

            # ------------------------------
            # Frequency / SRM branch
            # ------------------------------

            image_tensor = (
                pil_to_tensor(image)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(device)
            )

            frequency_embedding, residual_energy = (
                frequency(image_tensor)
            )

            # ------------------------------
            # Fusion
            # ------------------------------

            logit = fusion(
                spatial_embedding,
                frequency_embedding,
                residual_energy,
            )

            probability = torch.sigmoid(
                logit
            ).item()

            # ------------------------------
            # Store metadata
            # ------------------------------

            labels.append(
                int(row["label"])
            )

            probabilities.append(
                probability
            )

            conditions.append(
                row["condition"]
            )

            sources.append(
                row["source"]
            )

    labels = np.array(labels)
    probabilities = np.array(probabilities)
    conditions = np.array(conditions)
    sources = np.array(sources)

    # ========================================================
    # AUC by condition
    # ========================================================

    unique_conditions = sorted(
        set(conditions)
    )

    per_condition = {}
    unseen_condition = {}

    for condition in unique_conditions:

        mask = conditions == condition

        condition_labels = labels[mask]
        condition_probs = probabilities[mask]

        auc = roc_auc_score(
            condition_labels,
            condition_probs,
        )

        per_condition[condition] = {
            "auc": auc,
            "n": int(mask.sum()),
        }

        # ----------------------------------------------------
        # Held-out generator subset
        # ----------------------------------------------------

        unseen_mask = (
            mask
            & np.isin(
                sources,
                list(UNSEEN_GENERATOR_SOURCES),
            )
        )

        unseen_labels = labels[unseen_mask]
        unseen_probs = probabilities[unseen_mask]

        if len(set(unseen_labels)) > 1:

            unseen_auc = roc_auc_score(
                unseen_labels,
                unseen_probs,
            )

            unseen_condition[condition] = {
                "auc": unseen_auc,
                "n": int(unseen_mask.sum()),
            }

    # ========================================================
    # Final metrics
    # ========================================================

    clean_auc = per_condition[
        "clean"
    ]["auc"]

    robust_conditions = [
        condition
        for condition in unique_conditions
        if condition != "clean"
    ]

    avg_robust_auc = float(
        np.mean(
            [
                per_condition[condition]["auc"]
                for condition in robust_conditions
            ]
        )
    )

    avg_robust_drop = (
        clean_auc - avg_robust_auc
    )

    final_score = (
        0.5 * clean_auc
        + 0.5 * avg_robust_auc
    )

    # ========================================================
    # Print table
    # ========================================================

    print("\n")
    print(
        f"{'condition':<22}"
        f"{'AUC':>10}"
        f"{'n':>8}"
        f"{'unseen-gen AUC':>18}"
    )

    print("-" * 58)

    for condition in unique_conditions:

        result = per_condition[
            condition
        ]

        unseen = unseen_condition.get(
            condition
        )

        if unseen:
            unseen_text = (
                f"{unseen['auc']:.4f}"
            )
        else:
            unseen_text = "--"

        print(
            f"{condition:<22}"
            f"{result['auc']:>10.4f}"
            f"{result['n']:>8}"
            f"{unseen_text:>18}"
        )

    print("\n")
    print(
        f"clean AUC:        "
        f"{clean_auc:.4f}"
    )

    print(
        f"avg robust AUC:   "
        f"{avg_robust_auc:.4f}"
    )

    print(
        f"avg robust drop:  "
        f"{avg_robust_drop:.4f}"
    )

    print(
        f"final score:      "
        f"{final_score:.4f}"
    )

    # ========================================================
    # Save results
    # ========================================================

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_PATH,
        "w",
    ) as f:

        f.write(
            f"{'condition':<22}"
            f"{'AUC':>10}"
            f"{'n':>8}"
            f"{'unseen-gen AUC':>18}\n"
        )

        f.write("-" * 58 + "\n")

        for condition in unique_conditions:

            result = per_condition[
                condition
            ]

            unseen = unseen_condition.get(
                condition
            )

            if unseen:
                unseen_text = (
                    f"{unseen['auc']:.4f}"
                )
            else:
                unseen_text = "--"

            f.write(
                f"{condition:<22}"
                f"{result['auc']:>10.4f}"
                f"{result['n']:>8}"
                f"{unseen_text:>18}\n"
            )

        f.write("\n")

        f.write(
            f"clean AUC:        "
            f"{clean_auc:.4f}\n"
        )

        f.write(
            f"avg robust AUC:   "
            f"{avg_robust_auc:.4f}\n"
        )

        f.write(
            f"avg robust drop:  "
            f"{avg_robust_drop:.4f}\n"
        )

        f.write(
            f"final score:      "
            f"{final_score:.4f}\n"
        )

    print(
        f"\nSaved results to: "
        f"{RESULT_PATH}"
    )


if __name__ == "__main__":
    main()