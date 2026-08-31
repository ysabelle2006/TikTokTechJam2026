import csv
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from tiktoktechjam2026.models.frequency_stream import FrequencyStream


EVAL_MANIFEST = Path(
    "data/eval_manifest.csv"
)

CHECKPOINT = Path(
    "checkpoints/v2_residual_fusion_best.pt"
)


class SRMClassifier(nn.Module):
    """
    Simple classifier on top of the 128-d SRM embedding.

    This classifier is NOT trained separately here.

    Instead, we will use the SRM branch from the residual-fusion
    checkpoint and inspect whether its learned representation
    contains useful clean-test signal by training a tiny linear
    probe on the shared training split first.
    """

    def __init__(self):
        super().__init__()

        self.classifier = nn.Linear(
            128,
            1,
        )

    def forward(self, x):
        return self.classifier(x).squeeze(1)


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ============================================================
    # 1. Load clean shared evaluation rows
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
        f"\nClean evaluation images: "
        f"{len(rows)}"
    )

    # ============================================================
    # 2. Load SRM branch from residual-fusion checkpoint
    # ============================================================

    print(
        "Loading trained SRM branch..."
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    frequency = FrequencyStream(
        "srm"
    ).to(device)

    frequency.load_state_dict(
        checkpoint["frequency"]
    )

    frequency.eval()

    # ============================================================
    # IMPORTANT
    #
    # Residual fusion did not contain an independent SRM
    # classifier, so there is no existing "SRM probability"
    # we can directly evaluate.
    #
    # Therefore this script extracts SRM embeddings from the
    # 900 clean shared images and saves them for a small probe.
    # ============================================================

    embeddings = []
    labels = []

    with torch.no_grad():

        for row in tqdm(
            rows,
            desc="Extracting clean SRM embeddings",
        ):

            image = Image.open(
                row["transformed_path"]
            ).convert("RGB")

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

            embeddings.append(
                frequency_embedding
                .squeeze(0)
                .cpu()
            )

            labels.append(
                int(row["label"])
            )

    embeddings = torch.stack(
        embeddings
    )

    labels = torch.tensor(
        labels,
        dtype=torch.float32,
    )

    output_path = Path(
        "results/srm_shared_clean_embeddings.pt"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
        },
        output_path,
    )

    print()
    print(
        "================================"
    )
    print(
        "SRM CLEAN EMBEDDING EXTRACTION"
    )
    print(
        "================================"
    )
    print(
        f"Images:      {len(labels)}"
    )
    print(
        f"Embeddings:  {tuple(embeddings.shape)}"
    )
    print(
        f"REAL:        {(labels == 0).sum().item()}"
    )
    print(
        f"FAKE:        {(labels == 1).sum().item()}"
    )
    print(
        f"Saved to:    {output_path}"
    )
    print(
        "================================"
    )


if __name__ == "__main__":
    main()