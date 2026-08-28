from src.tiktoktechjam2026.data.datasets import AIGCFolderDataset
from src.tiktoktechjam2026.transforms.preprocessing import prepare_spatial_input
from src.tiktoktechjam2026.transforms.augmentations import random_transform


dataset = AIGCFolderDataset(
    root_dir="data/cifake/train",
    augmentation=random_transform,
    transform=prepare_spatial_input
)

print("Number of training images:", len(dataset))

for i in range(5):
    image, label = dataset[i]

    print(
        "Image", i,
        "| shape:", image.shape,
        "| label:", label
    )