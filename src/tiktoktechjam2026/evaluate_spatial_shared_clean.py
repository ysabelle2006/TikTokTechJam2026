import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from tiktoktechjam2026.models.spatial_stream import SpatialStream


EVAL_MANIFEST = Path("data/eval_manifest.csv")
CHECKPOINT = Path(
    "checkpoints/v0_spatial_sharedtrain_best.pt"
)


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


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------
    # Get only the 900 CLEAN shared-evaluation images
    # --------------------------------------------------
    with open(EVAL_MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))

    rows = [
        row for row in rows
        if row["condition"] == "clean"
    ]

    print("Clean evaluation images:", len(rows))

    # --------------------------------------------------
    # Load CLIP
    # --------------------------------------------------
    print("Loading spatial stream...")

    spatial = SpatialStream()
    spatial.model.eval()

    # --------------------------------------------------
    # Load trained spatial head
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

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    with torch.no_grad():

        for row in tqdm(
            rows,
            desc="Evaluating clean shared set",
        ):
            image_path = Path(
                row["transformed_path"]
            )

            image = Image.open(
                image_path
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

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    print()
    print("==============================")
    print("SPATIAL-ONLY SHARED CLEAN TEST")
    print("==============================")
    print(f"Images:    {len(rows)}")
    print(f"Clean AUC: {auc:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()