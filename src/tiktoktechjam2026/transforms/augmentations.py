"""
Train-time robustness augmentations.

Implements the exact transform families and parameter ranges from the
challenge brief (section 5.2), applied randomly during training so the
model learns to stay accurate after the same real-world post-processing
the organizers will test against:

    JPEG compression   quality in {90, 70, 50, 30}
    Gaussian blur      sigma in {0.5, 1.0, 2.0}
    Resize             scale 0.5x or 0.25x, then upscale back
    Gaussian noise     sigma in {0.02, 0.05, 0.10}
    Color jitter       brightness/contrast/saturation +/- 20%
    Center crop        crop to 80%, then resize back

TODO (next step): implement each function against a PIL.Image or numpy
array, plus a random_transform(image) dispatcher that applies zero or
more of these per call -- the brief tests "a subset" of these, not
necessarily all stacked together every time.
"""


import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------
# JPEG compression
# quality = 90, 70, 50, 30
# ---------------------------------------------------------
def jpeg_compression(image, quality):
    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    compressed = Image.open(buffer).convert("RGB")
    compressed.load()

    return compressed


# ---------------------------------------------------------
# Gaussian blur
# sigma = 0.5, 1.0, 2.0
# ---------------------------------------------------------
def gaussian_blur(image, sigma):
    return image.filter(
        ImageFilter.GaussianBlur(radius=sigma)
    )


# ---------------------------------------------------------
# Resize
# scale = 0.5 or 0.25, then upscale back
# ---------------------------------------------------------
def resize_transform(image, scale):
    original_width, original_height = image.size

    small_width = max(1, int(original_width * scale))
    small_height = max(1, int(original_height * scale))

    small = image.resize(
        (small_width, small_height),
        Image.Resampling.BILINEAR,
    )

    restored = small.resize(
        (original_width, original_height),
        Image.Resampling.BILINEAR,
    )

    return restored


# ---------------------------------------------------------
# Gaussian noise
# sigma = 0.02, 0.05, 0.10
#
# Pixel values are converted to [0, 1], noise is added,
# then converted back to uint8.
# ---------------------------------------------------------
def gaussian_noise(image, sigma, seed=42):
    array = np.asarray(image.convert("RGB")).astype(
        np.float32
    ) / 255.0

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=array.shape,
    )

    noisy = np.clip(array + noise, 0.0, 1.0)

    noisy = (noisy * 255).astype(np.uint8)

    return Image.fromarray(noisy)


# ---------------------------------------------------------
# Color jitter
# brightness / contrast / saturation +/- 20%
#
# factor:
# 0.8 = -20%
# 1.2 = +20%
# ---------------------------------------------------------
def color_jitter(image, factor):
    result = image.convert("RGB")

    result = ImageEnhance.Brightness(result).enhance(
        factor
    )

    result = ImageEnhance.Contrast(result).enhance(
        factor
    )

    result = ImageEnhance.Color(result).enhance(
        factor
    )

    return result


# ---------------------------------------------------------
# Center crop
# keep 80% of image, then resize back
# ---------------------------------------------------------
def center_crop(image, crop_fraction=0.8):
    original_width, original_height = image.size

    crop_width = int(original_width * crop_fraction)
    crop_height = int(original_height * crop_fraction)

    left = (original_width - crop_width) // 2
    top = (original_height - crop_height) // 2

    right = left + crop_width
    bottom = top + crop_height

    cropped = image.crop(
        (left, top, right, bottom)
    )

    restored = cropped.resize(
        (original_width, original_height),
        Image.Resampling.BILINEAR,
    )

    return restored


# ---------------------------------------------------------
# Exact evaluation grid from the challenge brief
# ---------------------------------------------------------
EVALUATION_TRANSFORMS = {
    "jpeg_90": lambda image: jpeg_compression(image, 90),
    "jpeg_70": lambda image: jpeg_compression(image, 70),
    "jpeg_50": lambda image: jpeg_compression(image, 50),
    "jpeg_30": lambda image: jpeg_compression(image, 30),

    "blur_0.5": lambda image: gaussian_blur(image, 0.5),
    "blur_1.0": lambda image: gaussian_blur(image, 1.0),
    "blur_2.0": lambda image: gaussian_blur(image, 2.0),

    "resize_0.5": lambda image: resize_transform(image, 0.5),
    "resize_0.25": lambda image: resize_transform(image, 0.25),

    "noise_0.02": lambda image: gaussian_noise(image, 0.02),
    "noise_0.05": lambda image: gaussian_noise(image, 0.05),
    "noise_0.10": lambda image: gaussian_noise(image, 0.10),

    "jitter_0.8": lambda image: color_jitter(image, 0.8),
    "jitter_1.2": lambda image: color_jitter(image, 1.2),

    "crop_0.8": lambda image: center_crop(image, 0.8),
}

STACKED_TRANSFORMS = {
    "jpeg30_resize025": lambda image:
        resize_transform(
            jpeg_compression(image, 30),
            0.25,
        ),

    "blur2_jpeg50": lambda image:
        jpeg_compression(
            gaussian_blur(image, 2.0),
            50,
        ),

    "resize025_noise010": lambda image:
        gaussian_noise(
            resize_transform(image, 0.25),
            0.10,
        ),

    "crop08_jpeg70": lambda image:
        jpeg_compression(
            center_crop(image, 0.8),
            70,
        ),

    "blur1_resize05_jpeg50": lambda image:
        jpeg_compression(
            resize_transform(
                gaussian_blur(image, 1.0),
                0.5,
            ),
            50,
        ),
}