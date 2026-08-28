from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class AIGCFolderDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        augmentation=None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.augmentation = augmentation

        self.samples = []

        class_map = {
            "REAL": 0,
            "FAKE": 1,
        }

        for class_name, label in class_map.items():
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp"
                }:
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        # Training augmentation happens first
        if self.augmentation is not None:
            image = self.augmentation(image)

        # Model preprocessing happens after
        if self.transform is not None:
            image = self.transform(image)

        return image, label