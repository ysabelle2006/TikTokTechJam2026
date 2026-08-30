import io
import random
from pathlib import Path

import numpy as np
import torch

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
)

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

START_CHECKPOINT = (
    "results/v2_fft_multidomain/best.pt"
)

OUTPUT_DIR = Path(
    "results/v3_transform_aware"
)

BEST_CHECKPOINT = (
    OUTPUT_DIR / "best.pt"
)

# Use a smaller CIFAKE subset so this
# experiment does not take forever.
CIFAKE_MAX_SAMPLES = 20000

BATCH_SIZE = 64

EPOCHS = 3

LEARNING_RATE = 5e-5

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

print(
    "Using device:",
    device,
)


# ==================================================
# Frequency preprocessing
# ==================================================

frequency_transform = (
    transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),
        transforms.ToTensor(),
    ])
)


# ==================================================
# Deterministic augmentation functions
# ==================================================

def jpeg_compress(
    image,
    quality,
):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    result = Image.open(
        buffer
    ).convert("RGB")

    return result


def gaussian_blur(
    image,
    sigma,
):

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=sigma
        )
    )


def resize_roundtrip(
    image,
    scale,
):

    width, height = image.size

    new_width = max(
        1,
        int(width * scale),
    )

    new_height = max(
        1,
        int(height * scale),
    )

    small = image.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.BILINEAR,
    )

    restored = small.resize(
        (
            width,
            height,
        ),
        Image.Resampling.BILINEAR,
    )

    return restored


def gaussian_noise(
    image,
    sigma,
    rng,
):

    array = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=array.shape,
    )

    array = (
        array + noise
    )

    array = np.clip(
        array,
        0.0,
        1.0,
    )

    array = (
        array * 255.0
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        array
    )


def deterministic_color_jitter(
    image,
    rng,
):

    # Moderate ±20%
    brightness = rng.uniform(
        0.8,
        1.2,
    )

    contrast = rng.uniform(
        0.8,
        1.2,
    )

    saturation = rng.uniform(
        0.8,
        1.2,
    )

    image = (
        ImageEnhance.Brightness(
            image
        ).enhance(
            brightness
        )
    )

    image = (
        ImageEnhance.Contrast(
            image
        ).enhance(
            contrast
        )
    )

    image = (
        ImageEnhance.Color(
            image
        ).enhance(
            saturation
        )
    )

    return image


def center_crop_80(
    image,
):

    width, height = image.size

    crop_width = int(
        width * 0.8
    )

    crop_height = int(
        height * 0.8
    )

    left = (
        width - crop_width
    ) // 2

    top = (
        height - crop_height
    ) // 2

    right = (
        left + crop_width
    )

    bottom = (
        top + crop_height
    )

    cropped = image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )

    # Restore original dimensions
    return cropped.resize(
        (
            width,
            height,
        ),
        Image.Resampling.BILINEAR,
    )


# ==================================================
# Apply one random moderate transform
# ==================================================
def apply_one_transform(
    image,
    rng,
):

    transform_id = int(
        rng.integers(
            0,
            6,
        )
    )

    # JPEG
    if transform_id == 0:

        quality = int(
            rng.choice([
                90,
                70,
                50,
            ])
        )

        return jpeg_compress(
            image,
            quality,
        )

    # Blur
    if transform_id == 1:

        sigma = float(
            rng.choice([
                0.5,
                1.0,
            ])
        )

        return gaussian_blur(
            image,
            sigma,
        )

    # Resize
    if transform_id == 2:

        scale = float(
            rng.choice([
                0.75,
                0.5,
            ])
        )

        return resize_roundtrip(
            image,
            scale,
        )

    # Noise
    if transform_id == 3:

        sigma = float(
            rng.choice([
                0.02,
                0.05,
            ])
        )

        return gaussian_noise(
            image,
            sigma,
            rng,
        )

    # Color jitter
    if transform_id == 4:

        return deterministic_color_jitter(
            image,
            rng,
        )

    # Center crop
    return center_crop_80(
        image
    )

   


# ==================================================
# Transform policy
#
# 30% clean
# 50% one transform
# 20% two transforms
# ==================================================

def transform_policy(
    image,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    probability = rng.random()

    # 30% clean
    if probability < 0.30:

        return image

    # 50% one transform
    if probability < 0.80:

        return apply_one_transform(
            image,
            rng,
        )

    # 20% two transforms
    image = apply_one_transform(
        image,
        rng,
    )

    image = apply_one_transform(
        image,
        rng,
    )

    return image


# ==================================================
# Dataset
# ==================================================

class TransformAwareDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        indices,
        domain,
        training=True,
    ):

        self.base_dataset = (
            base_dataset
        )

        self.indices = list(
            indices
        )

        self.domain = domain

        self.training = training

        self.epoch = 0

    def set_epoch(
        self,
        epoch,
    ):

        self.epoch = epoch

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

        image_path, label = (
            self.base_dataset.samples[
                original_index
            ]
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        # ==============================================
        # Training augmentation
        # ==============================================

        if self.training:

            augmentation_seed = (
                SEED
                + self.epoch * 1_000_000
                + self.domain * 100_000
                + original_index
            )

            image = transform_policy(
                image,
                augmentation_seed,
            )

        # ==============================================
        # IMPORTANT:
        # BOTH branches receive SAME transformed image
        # ==============================================

        spatial_image = (
            prepare_spatial_input(
                image
            )
        )

        frequency_image = (
            frequency_transform(
                image
            )
        )

        label = torch.tensor(
            float(label),
            dtype=torch.float32,
        )

        domain = torch.tensor(
            self.domain,
            dtype=torch.long,
        )

        return (
            spatial_image,
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
            self.lookup[
                index
            ]
        )

        return (
            self.datasets[
                dataset_id
            ][local_index]
        )


# ==================================================
# Validation
# ==================================================

def evaluate_validation(
    loader,
    spatial_stream,
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
            spatial_images,
            frequency_images,
            labels,
            domains,
        ) in loader:

            # ==========================================
            # Spatial
            # ==========================================

            spatial_embeddings = (
                spatial_stream.encode(
                    spatial_images
                )
            )

            # ==========================================
            # Frequency
            # ==========================================

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

            probabilities = (
                torch.sigmoid(
                    logits
                )
            )

            predictions = (
                probabilities
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

            # CIFAKE
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

            # SID
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

    print()
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    cifake_dataset = (
        AIGCFolderDataset(
            root_dir=CIFAKE_DIR,
            transform=None,
            augmentation=None,
        )
    )

    sid_dataset = (
        AIGCFolderDataset(
            root_dir=SID_DIR,
            transform=None,
            augmentation=None,
        )
    )

    print(
        "Full CIFAKE:",
        len(cifake_dataset),
    )

    print(
        "SID:",
        len(sid_dataset),
    )

    # ==================================================
    # Balanced CIFAKE subset
    # ==================================================

    cifake_real = []

    cifake_fake = []

    for index, (
        image_path,
        label,
    ) in enumerate(
        cifake_dataset.samples
    ):

        if label == 0:

            cifake_real.append(
                index
            )

        else:

            cifake_fake.append(
                index
            )

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        cifake_real
    )

    rng.shuffle(
        cifake_fake
    )

    half = (
        CIFAKE_MAX_SAMPLES
        // 2
    )

    cifake_selected = (
        cifake_real[:half]
        +
        cifake_fake[:half]
    )

    rng.shuffle(
        cifake_selected
    )

    print(
        "Selected CIFAKE:",
        len(cifake_selected),
    )

    # ==================================================
    # SID indices
    # ==================================================

    sid_indices = list(
        range(
            len(sid_dataset)
        )
    )

    rng.shuffle(
        sid_indices
    )

    # ==================================================
    # Train / validation split
    # ==================================================

    cifake_split = int(
        TRAIN_RATIO
        * len(cifake_selected)
    )

    sid_split = int(
        TRAIN_RATIO
        * len(sid_indices)
    )

    cifake_train_indices = (
        cifake_selected[
            :cifake_split
        ]
    )

    cifake_val_indices = (
        cifake_selected[
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
    # Dataset objects
    # ==================================================

    cifake_train = (
        TransformAwareDataset(
            cifake_dataset,
            cifake_train_indices,
            domain=0,
            training=True,
        )
    )

    sid_train = (
        TransformAwareDataset(
            sid_dataset,
            sid_train_indices,
            domain=1,
            training=True,
        )
    )

    # Validation stays CLEAN
    cifake_val = (
        TransformAwareDataset(
            cifake_dataset,
            cifake_val_indices,
            domain=0,
            training=False,
        )
    )

    sid_val = (
        TransformAwareDataset(
            sid_dataset,
            sid_val_indices,
            domain=1,
            training=False,
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
    # Equal domain sampling
    # ==================================================

    cifake_weight = (
        1.0
        / len(cifake_train)
    )

    sid_weight = (
        1.0
        / len(sid_train)
    )

    weights = (
        [cifake_weight]
        * len(cifake_train)
        +
        [sid_weight]
        * len(sid_train)
    )

    sampler = (
        WeightedRandomSampler(
            weights=weights,
            num_samples=len(
                train_dataset
            ),
            replacement=True,
            generator=(
                torch.Generator()
                .manual_seed(
                    SEED
                )
            ),
        )
    )

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
        )
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
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
    # Load V2 checkpoint
    # ==================================================

    print()
    print("=" * 60)
    print("LOADING V2 CHECKPOINT")
    print("=" * 60)

    checkpoint = torch.load(
        START_CHECKPOINT,
        map_location=device,
    )

    print(
        "Starting from:",
        START_CHECKPOINT,
    )

    print(
        "Previous val accuracy:",
        checkpoint[
            "val_accuracy"
        ],
    )

    # ==================================================
    # Spatial stream
    #
    # Frozen CLIP.
    # We DO NOT train CLIP.
    # ==================================================

    spatial_stream = (
        SpatialStream(
            freeze=True
        )
    )

    # ==================================================
    # FFT frequency stream
    # ==================================================

    frequency_stream = (
        FrequencyStream(
            mode="fft"
        ).to(device)
    )

    frequency_stream.load_state_dict(
        checkpoint[
            "frequency_stream"
        ]
    )

    # ==================================================
    # Fusion
    # ==================================================

    fusion_head = (
        FusionHead().to(
            device
        )
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

    optimizer = (
        torch.optim.Adam(
            parameters,
            lr=LEARNING_RATE,
        )
    )

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )

    # ==================================================
    # Training
    # ==================================================

    best_val_accuracy = 0.0

    print()
    print("=" * 60)
    print(
        "STARTING V3 TRANSFORM-AWARE TRAINING"
    )
    print("=" * 60)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        # Change deterministic augmentations
        # each epoch
        cifake_train.set_epoch(
            epoch
        )

        sid_train.set_epoch(
            epoch
        )

        frequency_stream.train()
        fusion_head.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_index, (
            spatial_images,
            frequency_images,
            labels,
            domains,
        ) in enumerate(
            train_loader
        ):

            # ==========================================
            # CLIP encoding of transformed images
            # ==========================================

            with torch.no_grad():

                spatial_embeddings = (
                    spatial_stream.encode(
                        spatial_images
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
                torch.sigmoid(
                    logits
                )
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
                % 50
                == 0
            ):

                print(
                    f"Epoch "
                    f"{epoch}/{EPOCHS} "
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
        # Clean validation
        # ==================================================

        (
            val_accuracy,
            cifake_val_accuracy,
            sid_val_accuracy,
        ) = evaluate_validation(
            val_loader,
            spatial_stream,
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
        # Save best checkpoint
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
                    "frequency_mode":
                        "fft",

                    "frequency_stream":
                        frequency_stream
                        .state_dict(),

                    "fusion_head":
                        fusion_head
                        .state_dict(),

                    "val_accuracy":
                        val_accuracy,

                    "cifake_val_accuracy":
                        cifake_val_accuracy,

                    "sid_val_accuracy":
                        sid_val_accuracy,

                    "epoch":
                        epoch,

                    "training_type":
                        (
                            "multidomain_"
                            "transform_aware"
                        ),
                },
                BEST_CHECKPOINT,
            )

            print(
                "✓ Saved new best checkpoint!"
            )

        print(
            "-" * 60
        )

    # ==================================================
    # Done
    # ==================================================

    print()
    print("=" * 60)
    print("V3 TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best validation accuracy:",
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        "Best checkpoint:",
        BEST_CHECKPOINT,
    )


if __name__ == "__main__":
    main()