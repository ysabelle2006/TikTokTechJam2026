from PIL import Image

from src.tiktoktechjam2026.data.datasets import AIGCFolderDataset
from src.tiktoktechjam2026.transforms.augmentations import random_transform


dataset = AIGCFolderDataset(
    root_dir="data/cifake/train",
    augmentation=None,
    transform=None
)

indices = [0, 1, 2, 50000, 50001]

for output_i, dataset_i in enumerate(indices):
    image_path, label = dataset.samples[dataset_i]

    image = Image.open(image_path).convert("RGB")
    augmented = random_transform(image)

    image.resize((256, 256)).save(
        f"visual_original_{output_i}.png"
    )
    augmented.resize((256, 256)).save(
        f"visual_augmented_{output_i}.png"
    )

    print(
        f"Dataset image {dataset_i} | label: {label} | "
        f"original size: {image.size} | "
        f"augmented size: {augmented.size}"
    )