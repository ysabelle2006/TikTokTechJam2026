import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.models.frequency_stream import FrequencyStream
from tiktoktechjam2026.models.fusion import ResidualFusionHead


# ============================================================
# Paths
# ============================================================

EVAL_MANIFEST = Path(
    "data/eval_manifest.csv"
)

SPATIAL_CHECKPOINT = Path(
    "checkpoints/v0_spatial_sharedtrain_best.pt"
)

RESIDUAL_CHECKPOINT = Path(
    "checkpoints/v2_residual_fusion_best.pt"
)


# ============================================================
# Spatial classifier
# Must match train_spatial_shared.py exactly
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # 1. Load ONLY clean shared-eval rows
    # --------------------------------------------------------

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
        f"\nClean evaluation images: "
        f"{len(rows)}"
    )

    # --------------------------------------------------------
    # 2. Load frozen CLIP spatial stream
    # --------------------------------------------------------

    print(
        "Loading spatial stream..."
    )

    spatial = SpatialStream()

    spatial.model.eval()

    for parameter in spatial.model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # 3. Load GOOD frozen spatial classifier
    # --------------------------------------------------------

    print(
        "Loading frozen spatial classifier..."
    )

    spatial_head = SpatialHead().to(
        device
    )

    spatial_head.load_state_dict(
        torch.load(
            SPATIAL_CHECKPOINT,
            map_location=device,
        )
    )

    spatial_head.eval()

    for parameter in spatial_head.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # 4. Load residual-fusion checkpoint
    # --------------------------------------------------------

    print(
        "Loading residual fusion checkpoint..."
    )

    checkpoint = torch.load(
        RESIDUAL_CHECKPOINT,
        map_location=device,
    )

    print(
        "Best training epoch:",
        checkpoint.get(
            "best_epoch",
            "?",
        ),
    )

    print(
        "Best validation loss:",
        checkpoint.get(
            "best_val_loss",
            "?",
        ),
    )

    # --------------------------------------------------------
    # 5. Load trained SRM
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 6. Load residual fusion head
    # --------------------------------------------------------

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
        f"Learned alpha: "
        f"{fusion.alpha.item():.4f}"
    )

    # --------------------------------------------------------
    # 7. Evaluate
    # --------------------------------------------------------

    labels = []
    probabilities = []
    spatial_probabilities = []

    with torch.no_grad():

        for row in tqdm(
            rows,
            desc="Evaluating residual fusion clean set",
        ):

            image = Image.open(
                row["transformed_path"]
            ).convert("RGB")

            # =================================================
            # Spatial branch
            # =================================================

            spatial_tensor = (
                spatial.preprocess(
                    image
                )
                .unsqueeze(0)
            )

            spatial_embedding = (
                spatial.encode(
                    spatial_tensor
                )
            )

            spatial_embedding = F.normalize(
                spatial_embedding.float(),
                dim=1,
            ).to(device)

            spatial_logit = (
                spatial_head(
                    spatial_embedding
                )
                .unsqueeze(1)
            )

            # =================================================
            # SRM branch
            # =================================================

            image_tensor = (
                pil_to_tensor(
                    image
                )
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

            # =================================================
            # Residual fusion
            # =================================================

            # Residual-fusion prediction
            final_logit = fusion(
                spatial_logit,
                frequency_embedding,
                residual_energy,
                )

# Spatial-only prediction from the EXACT SAME pass
            spatial_probability = torch.sigmoid(
                spatial_logit
                ).item()

            residual_probability = torch.sigmoid(
                final_logit
            ).item()

            spatial_probabilities.append(
                spatial_probability
            )

            probabilities.append(
                residual_probability
            )

            labels.append(
                int(row["label"])
            )


    # --------------------------------------------------------
    # 8. Clean AUC
    # --------------------------------------------------------

    spatial_auc = roc_auc_score(
        labels,
        spatial_probabilities,
    )

    residual_auc = roc_auc_score(
        labels,
        probabilities,
    )

    print()
    print("================================")
    print("SAME-PASS CLEAN DIAGNOSTIC")
    print("================================")
    print(f"Images:               {len(rows)}")
    print(f"Spatial-only AUC:      {spatial_auc:.4f}")
    print(f"Residual-fusion AUC:   {residual_auc:.4f}")
    print(f"Learned alpha:         {fusion.alpha.item():.4f}")
    print("================================")

    print()
    print(
        "================================"
    )
    print(
        "RESIDUAL FUSION SHARED CLEAN TEST"
    )
    print(
        "================================"
    )

    print(
        f"Images:    {len(rows)}"
    )

    print(
        f"Clean AUC: {auc:.4f}"
    )

    print(
        f"Alpha:     "
        f"{fusion.alpha.item():.4f}"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()