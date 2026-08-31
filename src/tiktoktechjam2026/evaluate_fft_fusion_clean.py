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


EVAL_MANIFEST = Path(
    "data/eval_manifest.csv"
)

SPATIAL_CHECKPOINT = Path(
    "checkpoints/v0_spatial_sharedtrain_best.pt"
)

FFT_CHECKPOINT = Path(
    "checkpoints/v2_residual_fusion_fft_best.pt"
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

    print("Device:", device)

    # --------------------------------------------------------
    # 1. Load only clean shared-eval rows
    # --------------------------------------------------------

    with open(
        EVAL_MANIFEST,
        newline="",
    ) as f:
        all_rows = list(csv.DictReader(f))

    rows = [
        row
        for row in all_rows
        if row["condition"] == "clean"
    ]

    print(
        f"\nClean evaluation images: {len(rows)}"
    )

    # --------------------------------------------------------
    # 2. Load spatial stream
    # --------------------------------------------------------

    print(
        "Loading spatial stream..."
    )

    spatial = SpatialStream()

    spatial.model.eval()

    for parameter in spatial.model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # 3. Load frozen spatial classifier
    # --------------------------------------------------------

    print(
        "Loading spatial classifier..."
    )

    spatial_head = SpatialHead().to(device)

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
    # 4. Load FFT residual checkpoint
    # --------------------------------------------------------

    print(
        "Loading FFT residual checkpoint..."
    )

    checkpoint = torch.load(
        FFT_CHECKPOINT,
        map_location=device,
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get("best_epoch", "?"),
    )

    print(
        "Checkpoint validation loss:",
        checkpoint.get("best_val_loss", "?"),
    )

    # --------------------------------------------------------
    # 5. Load trained FFT branch
    # --------------------------------------------------------

    print(
        "Loading trained FFT..."
    )

    frequency = FrequencyStream(
        "fft"
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

    fusion = ResidualFusionHead().to(device)

    fusion.load_state_dict(
        checkpoint["fusion"]
    )

    fusion.eval()

    print(
        f"Learned alpha: {fusion.alpha.item():.4f}"
    )

    # --------------------------------------------------------
    # 7. Evaluate
    # --------------------------------------------------------

    labels = []

    spatial_probabilities = []
    fft_fusion_probabilities = []

    with torch.no_grad():

        for row in tqdm(
            rows,
            desc="Evaluating spatial + FFT clean set",
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

            spatial_embedding = spatial.encode(
                spatial_tensor
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
            # FFT branch
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

            final_logit = fusion(
                spatial_logit,
                frequency_embedding,
                residual_energy,
            )

            spatial_probability = torch.sigmoid(
                spatial_logit
            ).item()

            fft_fusion_probability = torch.sigmoid(
                final_logit
            ).item()

            labels.append(
                int(row["label"])
            )

            spatial_probabilities.append(
                spatial_probability
            )

            fft_fusion_probabilities.append(
                fft_fusion_probability
            )

    # --------------------------------------------------------
    # 8. AUC comparison
    # --------------------------------------------------------

    spatial_auc = roc_auc_score(
        labels,
        spatial_probabilities,
    )

    fft_fusion_auc = roc_auc_score(
        labels,
        fft_fusion_probabilities,
    )

    improvement = (
        fft_fusion_auc
        - spatial_auc
    )

    print()
    print(
        "===================================="
    )
    print(
        "SPATIAL + FFT CLEAN DIAGNOSTIC"
    )
    print(
        "===================================="
    )

    print(
        f"Images:              {len(rows)}"
    )

    print(
        f"Spatial-only AUC:     {spatial_auc:.4f}"
    )

    print(
        f"Spatial + FFT AUC:    {fft_fusion_auc:.4f}"
    )

    print(
        f"AUC change:           {improvement:+.4f}"
    )

    print(
        f"Learned alpha:        {fusion.alpha.item():.4f}"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()