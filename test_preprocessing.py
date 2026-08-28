from PIL import Image

from src.tiktoktechjam2026.transforms.preprocessing import prepare_spatial_input


image = Image.open("test_image.jpg")

processed_image = prepare_spatial_input(image)

print("Original image size:", image.size)
print("Processed shape:", processed_image.shape)
print("Data type:", processed_image.dtype)