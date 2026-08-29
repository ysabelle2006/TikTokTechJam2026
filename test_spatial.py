from tiktoktechjam2026.data.datasets import CIFAKEDataset
from tiktoktechjam2026.models.spatial_stream import SpatialStream

dataset = CIFAKEDataset("data/CIFAKE", split="train")

image, label = dataset[0]

spatial = SpatialStream()

image_tensor = spatial.preprocess(image).unsqueeze(0)

embedding = spatial.encode(image_tensor)

print("Label:", label)
print("Tensor shape:", image_tensor.shape)
print("Embedding shape:", embedding.shape)