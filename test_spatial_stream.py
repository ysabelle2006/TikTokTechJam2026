from PIL import Image

from src.tiktoktechjam2026.models.spatial_stream import SpatialStream
from src.tiktoktechjam2026.transforms.preprocessing import prepare_spatial_input


image = Image.open("test_image.jpg").convert("RGB")

image_tensor = prepare_spatial_input(image)

model = SpatialStream()

embedding = model.encode(image_tensor)

print("Embedding shape:", embedding.shape)
print("Device:", embedding.device)