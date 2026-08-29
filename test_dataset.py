from tiktoktechjam2026.data.datasets import CIFAKEDataset

dataset = CIFAKEDataset("data/CIFAKE", split="train")

print("Number of images:", len(dataset))

image, label = dataset[0]

print("Image type:", type(image))
print("Label:", label)