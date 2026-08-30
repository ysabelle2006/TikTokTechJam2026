import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torchvision import transforms

from tiktoktechjam2026.data.datasets import AIGCFolderDataset
from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.models.frequency_stream import FrequencyStream
from tiktoktechjam2026.models.fusion import FusionHead
from tiktoktechjam2026.transforms.preprocessing import prepare_spatial_input


# ==================================================
# Configuration
# ==================================================

SEED = 42

CIFAKE_DIR = "data/cifake/train"
SID_DIR = "data/sid_train_subset"

CIFAKE_CACHE = "results/v0/train_embeddings_full.pt"
SID_CACHE = "results/v2_fft_multidomain/sid_train_embeddings.pt"

START_CHECKPOINT = "results/v1_fft/best.pt"

OUTPUT_DIR = Path("results/v2_fft_multidomain")
BEST_CHECKPOINT = OUTPUT_DIR / "best.pt"

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4

TRAIN_RATIO = 0.9


# ==================================================
# Reproducibility
# ==================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# Device
# ==================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# ==================================================
# Frequency preprocessing
# ==================================================

frequency_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ==================================================
# Cache SID CLIP embeddings
# ==================================================

def create_sid_cache():

    print()
    print("=" * 60)
    print("CREATING SID SPATIAL EMBEDDING CACHE")
    print("=" * 60)

    dataset = AIGCFolderDataset(
        root_dir=SID_DIR,
        augmentation=None,
        transform=None,
    )

    print(
        f"SID images found: {len(dataset)}"
    )

    spatial_stream = SpatialStream(
        freeze=True
    )

    all_embeddings = []
    all_labels = []

    batch_size = 64

    for start in range(
        0,
        len(dataset),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(dataset),
        )

        images = []
        labels = []

        for index in range(start, end):

            image_path, label = (
                dataset.samples[index]
            )

            from PIL import Image

            image = Image.open(
                image_path
            ).convert("RGB")

            image_tensor = (
                prepare_spatial_input(image)
            )

            images.append(
                image_tensor
            )

            labels.append(
                label
            )

        image_batch = torch.stack(
            images
        )

        with torch.no_grad():

            embeddings = (
                spatial_stream.encode(
                    image_batch
                )
            )

        all_embeddings.append(
            embeddings.cpu()
        )

        all_labels.extend(
            labels
        )

        print(
            f"\rCached {end}/{len(dataset)}",
            end="",
        )

    print()

    embeddings = torch.cat(
        all_embeddings,
        dim=0,
    )

    labels = torch.tensor(
        all_labels,
        dtype=torch.long,
    )

    Path(SID_CACHE).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
        },
        SID_CACHE,
    )

    print(
        "SID cache saved to:",
        SID_CACHE,
    )

    print(
        "Embeddings:",
        embeddings.shape,
    )

    print(
        "Labels:",
        labels.shape,
    )


# ==================================================
# Load cache helper
# ==================================================

def load_cache(path):

    cache = torch.load(
        path,
        map_location="cpu",
    )

    # Supports either:
    #
    # {
    #   "embeddings": tensor,
    #   "labels": tensor
    # }
    #
    # OR old tuple/list format

    if isinstance(cache, dict):

        embeddings = cache[
            "embeddings"
        ]

        labels = cache[
            "labels"
        ]

    else:

        embeddings = cache[0]
        labels = cache[1]

    return embeddings, labels


# ==================================================
# Dataset that pairs cached CLIP features
# with the raw image for FFT
# ==================================================

class CachedMultiDomainDataset(Dataset):

    def __init__(
        self,
        image_dataset,
        embeddings,
        labels,
        indices,
        domain,
    ):

        self.image_dataset = (
            image_dataset
        )

        self.embeddings = embeddings
        self.labels = labels
        self.indices = list(indices)

        self.domain = domain

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        dataset_index,
    ):

        original_index = (
            self.indices[
                dataset_index
            ]
        )

        image_path, image_label = (
            self.image_dataset.samples[
                original_index
            ]
        )

        from PIL import Image

        image = Image.open(
            image_path
        ).convert("RGB")

        frequency_image = (
            frequency_transform(
                image
            )
        )

        spatial_embedding = (
            self.embeddings[
                original_index
            ]
        )

        cached_label = int(
            self.labels[
                original_index
            ]
        )

        # Safety check:
        # cached embeddings must match
        # the raw image ordering
        if cached_label != image_label:

            raise RuntimeError(
                "Cache/image label mismatch at "
                f"{image_path}. "
                "Your cache may not match "
                "the current dataset ordering."
            )

        label = torch.tensor(
            float(image_label),
            dtype=torch.float32,
        )

        # Domain:
        # 0 = CIFAKE
        # 1 = SID

        domain = torch.tensor(
            self.domain,
            dtype=torch.long,
        )

        return (
            spatial_embedding,
            frequency_image,
            label,
            domain,
        )


# ==================================================
# Combined dataset
# ==================================================

class CombinedDataset(Dataset):

    def __init__(
        self,
        datasets,
    ):

        self.datasets = datasets

        self.lookup = []

        for dataset_id, dataset in enumerate(
            datasets
        ):

            for local_index in range(
                len(dataset)
            ):

                self.lookup.append(
                    (
                        dataset_id,
                        local_index,
                    )
                )

    def __len__(self):

        return len(
            self.lookup
        )

    def __getitem__(
        self,
        index,
    ):

        dataset_id, local_index = (
            self.lookup[index]
        )

        return self.datasets[
            dataset_id
        ][local_index]


# ==================================================
# Accuracy helper
# ==================================================

def calculate_accuracy(
    logits,
    labels,
):

    predictions = (
        torch.sigmoid(logits)
        >= 0.5
    ).float()

    return (
        predictions
        == labels
    ).float().mean().item()


# ==================================================
# Validation
# ==================================================

def evaluate_validation(
    loader,
    frequency_stream,
    fusion_head,
):

    frequency_stream.eval()
    fusion_head.eval()

    total_correct = 0
    total_samples = 0

    cifake_correct = 0
    cifake_total = 0

    sid_correct = 0
    sid_total = 0

    with torch.no_grad():

        for (
            spatial_embeddings,
            frequency_images,
            labels,
            domains,
        ) in loader:

            spatial_embeddings = (
                spatial_embeddings.to(
                    device
                )
            )

            frequency_images = (
                frequency_images.to(
                    device
                )
            )

            labels = labels.to(
                device
            )

            domains = domains.to(
                device
            )

            (
                frequency_embeddings,
                residual_energy,
            ) = frequency_stream(
                frequency_images
            )

            logits = fusion_head(
                spatial_embeddings,
                frequency_embeddings,
                residual_energy,
            ).squeeze(1)

            predictions = (
                torch.sigmoid(logits)
                >= 0.5
            ).float()

            correct = (
                predictions
                == labels
            )

            total_correct += (
                correct.sum().item()
            )

            total_samples += (
                labels.numel()
            )

            # CIFAKE domain
            cifake_mask = (
                domains == 0
            )

            if cifake_mask.any():

                cifake_correct += (
                    correct[
                        cifake_mask
                    ].sum().item()
                )

                cifake_total += (
                    cifake_mask.sum().item()
                )

            # SID domain
            sid_mask = (
                domains == 1
            )

            if sid_mask.any():

                sid_correct += (
                    correct[
                        sid_mask
                    ].sum().item()
                )

                sid_total += (
                    sid_mask.sum().item()
                )

    overall_accuracy = (
        total_correct
        / total_samples
    )

    cifake_accuracy = (
        cifake_correct
        / cifake_total
        if cifake_total > 0
        else 0.0
    )

    sid_accuracy = (
        sid_correct
        / sid_total
        if sid_total > 0
        else 0.0
    )

    return (
        overall_accuracy,
        cifake_accuracy,
        sid_accuracy,
    )


# ==================================================
# Main
# ==================================================

def main():

    # ==================================================
    # Make SID embedding cache if needed
    # ==================================================

    if not Path(
        SID_CACHE
    ).exists():

        create_sid_cache()

    else:

        print(
            "Using existing SID cache:",
            SID_CACHE,
        )

    # ==================================================
    # Load image datasets
    # ==================================================

    print()
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    cifake_images = (
        AIGCFolderDataset(
            root_dir=CIFAKE_DIR,
            augmentation=None,
            transform=None,
        )
    )

    sid_images = (
        AIGCFolderDataset(
            root_dir=SID_DIR,
            augmentation=None,
            transform=None,
        )
    )

    print(
        "CIFAKE images:",
        len(cifake_images),
    )

    print(
        "SID images:",
        len(sid_images),
    )

    # ==================================================
    # Load cached CLIP embeddings
    # ==================================================

    (
        cifake_embeddings,
        cifake_labels,
    ) = load_cache(
        CIFAKE_CACHE
    )

    (
        sid_embeddings,
        sid_labels,
    ) = load_cache(
        SID_CACHE
    )

    print()
    print(
        "CIFAKE embeddings:",
        cifake_embeddings.shape,
    )

    print(
        "SID embeddings:",
        sid_embeddings.shape,
    )

    # ==================================================
    # Safety checks
    # ==================================================

    assert (
        len(cifake_images)
        == len(cifake_embeddings)
    ), (
        "CIFAKE image/cache "
        "length mismatch"
    )

    assert (
        len(sid_images)
        == len(sid_embeddings)
    ), (
        "SID image/cache "
        "length mismatch"
    )

    # ==================================================
    # Separate train/validation split
    # for each domain
    # ==================================================

    generator = torch.Generator()
    generator.manual_seed(SEED)

    cifake_indices = torch.randperm(
        len(cifake_images),
        generator=generator,
    ).tolist()

    sid_indices = torch.randperm(
        len(sid_images),
        generator=generator,
    ).tolist()

    cifake_split = int(
        TRAIN_RATIO
        * len(cifake_indices)
    )

    sid_split = int(
        TRAIN_RATIO
        * len(sid_indices)
    )

    cifake_train_indices = (
        cifake_indices[
            :cifake_split
        ]
    )

    cifake_val_indices = (
        cifake_indices[
            cifake_split:
        ]
    )

    sid_train_indices = (
        sid_indices[
            :sid_split
        ]
    )

    sid_val_indices = (
        sid_indices[
            sid_split:
        ]
    )

    print()
    print("=" * 60)
    print("SPLIT")
    print("=" * 60)

    print(
        "CIFAKE train:",
        len(cifake_train_indices),
    )

    print(
        "CIFAKE val:",
        len(cifake_val_indices),
    )

    print(
        "SID train:",
        len(sid_train_indices),
    )

    print(
        "SID val:",
        len(sid_val_indices),
    )

    # ==================================================
    # Create domain datasets
    # ==================================================

    cifake_train = (
        CachedMultiDomainDataset(
            cifake_images,
            cifake_embeddings,
            cifake_labels,
            cifake_train_indices,
            domain=0,
        )
    )

    sid_train = (
        CachedMultiDomainDataset(
            sid_images,
            sid_embeddings,
            sid_labels,
            sid_train_indices,
            domain=1,
        )
    )

    cifake_val = (
        CachedMultiDomainDataset(
            cifake_images,
            cifake_embeddings,
            cifake_labels,
            cifake_val_indices,
            domain=0,
        )
    )

    sid_val = (
        CachedMultiDomainDataset(
            sid_images,
            sid_embeddings,
            sid_labels,
            sid_val_indices,
            domain=1,
        )
    )

    train_dataset = (
        CombinedDataset([
            cifake_train,
            sid_train,
        ])
    )

    val_dataset = (
        CombinedDataset([
            cifake_val,
            sid_val,
        ])
    )

    # ==================================================
    # Weighted sampling
    #
    # CIFAKE has ~90k train images
    # SID only ~5.4k.
    #
    # Without weighting, SID would barely
    # affect training.
    #
    # Give each DOMAIN approximately
    # equal sampling probability.
    # ==================================================

    cifake_weight = (
        1.0
        / len(cifake_train)
    )

    sid_weight = (
        1.0
        / len(sid_train)
    )

    sample_weights = []

    for _ in range(
        len(cifake_train)
    ):

        sample_weights.append(
            cifake_weight
        )

    for _ in range(
        len(sid_train)
    ):

        sample_weights.append(
            sid_weight
        )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(
            train_dataset
        ),
        replacement=True,
        generator=torch.Generator().manual_seed(
            SEED
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        "Training samples per epoch:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    # ==================================================
    # Load V1 FFT checkpoint
    # ==================================================

    print()
    print("=" * 60)
    print("LOADING V1 FFT CHECKPOINT")
    print("=" * 60)

    checkpoint = torch.load(
        START_CHECKPOINT,
        map_location=device,
    )

    print(
        "Starting checkpoint:",
        START_CHECKPOINT,
    )

    print(
        "Previous validation accuracy:",
        checkpoint[
            "val_accuracy"
        ],
    )

    frequency_stream = (
        FrequencyStream(
            mode="fft"
        ).to(device)
    )

    fusion_head = (
        FusionHead().to(
            device
        )
    )

    frequency_stream.load_state_dict(
        checkpoint[
            "frequency_stream"
        ]
    )

    fusion_head.load_state_dict(
        checkpoint[
            "fusion_head"
        ]
    )

    # ==================================================
    # Optimizer
    # ==================================================

    parameters = (
        list(
            frequency_stream.parameters()
        )
        +
        list(
            fusion_head.parameters()
        )
    )

    optimizer = torch.optim.Adam(
        parameters,
        lr=LEARNING_RATE,
    )

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )

    # ==================================================
    # Training
    # ==================================================

    print()
    print("=" * 60)
    print("STARTING MULTI-DOMAIN FFT TRAINING")
    print("=" * 60)

    best_val_accuracy = 0.0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        frequency_stream.train()
        fusion_head.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_index, (
            spatial_embeddings,
            frequency_images,
            labels,
            domains,
        ) in enumerate(
            train_loader
        ):

            spatial_embeddings = (
                spatial_embeddings.to(
                    device
                )
            )

            frequency_images = (
                frequency_images.to(
                    device
                )
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            (
                frequency_embeddings,
                residual_energy,
            ) = frequency_stream(
                frequency_images
            )

            logits = fusion_head(
                spatial_embeddings,
                frequency_embeddings,
                residual_energy,
            ).squeeze(1)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = (
                torch.sigmoid(logits)
                >= 0.5
            ).float()

            running_correct += (
                (
                    predictions
                    == labels
                )
                .sum()
                .item()
            )

            running_total += (
                labels.size(0)
            )

            if (
                (batch_index + 1)
                % 200
                == 0
            ):

                print(
                    f"Epoch {epoch}/{EPOCHS} "
                    f"| Batch "
                    f"{batch_index + 1}/"
                    f"{len(train_loader)} "
                    f"| Loss "
                    f"{loss.item():.4f}"
                )

        train_loss = (
            running_loss
            / running_total
        )

        train_accuracy = (
            running_correct
            / running_total
        )

        # ==================================================
        # Validation
        # ==================================================

        (
            val_accuracy,
            cifake_val_accuracy,
            sid_val_accuracy,
        ) = evaluate_validation(
            val_loader,
            frequency_stream,
            fusion_head,
        )

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train loss:        "
            f"{train_loss:.4f}"
        )

        print(
            f"Train accuracy:    "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Overall val acc:   "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"CIFAKE val acc:    "
            f"{cifake_val_accuracy * 100:.2f}%"
        )

        print(
            f"SID val acc:       "
            f"{sid_val_accuracy * 100:.2f}%"
        )

        # ==================================================
        # Save best model
        # ==================================================

        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )

            torch.save(
                {
                    "frequency_mode": "fft",

                    "frequency_stream":
                        frequency_stream.state_dict(),

                    "fusion_head":
                        fusion_head.state_dict(),

                    "val_accuracy":
                        val_accuracy,

                    "cifake_val_accuracy":
                        cifake_val_accuracy,

                    "sid_val_accuracy":
                        sid_val_accuracy,

                    "epoch":
                        epoch,

                    "training_type":
                        "cifake_sid_multidomain",
                },
                BEST_CHECKPOINT,
            )

            print(
                "✓ Saved new best "
                "checkpoint!"
            )

        print(
            "-" * 60
        )

    # ==================================================
    # Done
    # ==================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best overall "
        f"validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        "Best checkpoint:",
        BEST_CHECKPOINT,
    )


if __name__ == "__main__":
    main()