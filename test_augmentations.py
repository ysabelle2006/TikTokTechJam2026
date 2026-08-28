from PIL import Image

from src.tiktoktechjam2026.transforms.augmentations import (
    jpeg_compress,
    gaussian_blur,
    resize_roundtrip,
    gaussian_noise,
    color_jitter,
    center_crop,
)


image = Image.open("test_image.jpg").convert("RGB")

jpeg_compress(image, 30).save("test_jpeg.jpg")
gaussian_blur(image, 2.0).save("test_blur.jpg")
resize_roundtrip(image, 0.25).save("test_resize.jpg")
gaussian_noise(image, 0.10).save("test_noise.jpg")
color_jitter(image).save("test_color.jpg")
center_crop(image, 0.8).save("test_crop.jpg")

print("Done! Six transformed images were created.")