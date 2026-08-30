import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm

from tiktoktechjam2026.models.spatial_stream import SpatialStream


EVAL_MANIFEST = Path("data/eval_manifest.csv")
CHECKPOINT = Path(
    "checkpoints/v0_spatial_sharedtrain_best.pt"
)

RESULT_PATH = Path(
    "results/v0_spatial_sharedtrain_robustness.txt"
)

UNSEEN_GENERATOR_SOURCES = {
    "coco_val2017",
    "wildfake_dalle",
}


class SpatialHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_eval_manifest():
    with open(
        EVAL_MANIFEST,
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    rows = load_eval_manifest()

    print(
        f"\nLoaded {len(rows)} evaluation rows."
    )

    condition_counts = Counter(
        row["condition"]
        for row in rows
    )

    print("\nCondition counts:")

    for condition, count in sorted(
        condition_counts.items()
    ):
        print(
            f"  {condition:<20} {count}"
        )

    # --------------------------------------------------
    # Load CLIP
    # --------------------------------------------------

    print("\nLoading spatial stream...")

    spatial = SpatialStream()
    spatial.model.eval()

    for parameter in spatial.model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------
    # Load spatial classifier
    # --------------------------------------------------

    print("Loading spatial classifier...")

    classifier = SpatialHead().to(device)

    classifier.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=device,
        )
    )

    classifier.eval()

    labels = []
    probabilities = []
    conditions = []
    sources = []

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    print(
        f"\nEvaluating {len(rows)} transformed images..."
    )

    with torch.no_grad():

        for row in tqdm(rows):

            image = Image.open(
                row["transformed_path"]
            ).convert("RGB")

            image_tensor = (
                spatial.preprocess(image)
                .unsqueeze(0)
            )

            embedding = spatial.encode(
                image_tensor
            )

            embedding = F.normalize(
                embedding.float(),
                dim=1,
            ).to(device)

            logit = classifier(
                embedding
            )

            probability = torch.sigmoid(
                logit
            ).item()

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

    # --------------------------------------------------
    # AUC by condition
    # --------------------------------------------------

    unique_conditions = sorted(
        set(conditions)
    )

    per_condition = {}
    unseen_condition = {}

    for condition in unique_conditions:

        mask = (
            conditions == condition
        )

        auc = roc_auc_score(
            labels[mask],
            probabilities[mask],
        )

        per_condition[condition] = {
            "auc": auc,
            "n": int(mask.sum()),
        }

        unseen_mask = (
            mask
            & np.isin(
                sources,
                list(
                    UNSEEN_GENERATOR_SOURCES
                ),
            )
        )

        unseen_labels = labels[
            unseen_mask
        ]

        unseen_probs = probabilities[
            unseen_mask
        ]

        if len(
            set(unseen_labels)
        ) > 1:

            unseen_auc = roc_auc_score(
                unseen_labels,
                unseen_probs,
            )

            unseen_condition[
                condition
            ] = {
                "auc": unseen_auc,
                "n": int(
                    unseen_mask.sum()
                ),
            }

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

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
                per_condition[
                    condition
                ]["auc"]
                for condition
                in robust_conditions
            ]
        )
    )

    avg_robust_drop = (
        clean_auc
        - avg_robust_auc
    )

    final_score = (
        0.5 * clean_auc
        + 0.5 * avg_robust_auc
    )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

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

        unseen_text = (
            f"{unseen['auc']:.4f}"
            if unseen
            else "--"
        )

        print(
            f"{condition:<22}"
            f"{result['auc']:>10.4f}"
            f"{result['n']:>8}"
            f"{unseen_text:>18}"
        )

    print()
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

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

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

        f.write(
            "-" * 58 + "\n"
        )

        for condition in unique_conditions:

            result = per_condition[
                condition
            ]

            unseen = unseen_condition.get(
                condition
            )

            unseen_text = (
                f"{unseen['auc']:.4f}"
                if unseen
                else "--"
            )

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