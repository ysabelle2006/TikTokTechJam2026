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
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def jpeg_compress(image, quality: int):
    """Simulate JPEG compression"""
    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    compressed = Image.open(buffer).convert("RGB")
    compressed.load()
    return compressed 
    


def gaussian_blur(image, sigma: float):
    """Apply Gaussian blur"""
    return image.filter(
        ImageFilter.GaussianBlur(radius=sigma)
    )


def resize_roundtrip(image, scale: float):
    """Shrink image, resize back to original size"""
    original_width, original_height = image.size

    new_width = max(1, int(original_width * scale))
    new_height = max(1, int(original_height * scale))

    smaller = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    restored = smaller.resize((original_width, original_height), Image.Resampling.BILINEAR)
    return restored

def gaussian_noise(image, sigma: float):
    """Add Gaussian noise to the image"""
    array = np.asarray(image.convert("RGB")).astype(np.float32)/255.0

    noise = np.random.normal(
        loc = 0.0,
        scale = sigma,
        size = array.shape
    )

    noisy = np.clip(array + noise, 0.0, 1.0)
    noisy = (noisy * 255).astype(np.uint8)
    return Image.fromarray(noisy)

def color_jitter(image, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2):
    image, 
    brightness = 0.2,
    contrast = 0.2,
    saturation = 0.2

    brightness_factor = random.uniform(0.8, 1.2)

    contrast_factor = random.uniform(0.8, 1.2)

    saturation_factor = random.uniform(0.8, 1.2)

    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    image = ImageEnhance.Color(image).enhance(saturation_factor)

    return image

def center_crop(image, crop_fraction: float = 0.8):
    width, height = image.size

    crop_width = int(width * crop_fraction)
    crop_height = int(height * crop_fraction)

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    cropped = image.crop(
        (left, top, right, bottom)
    )

    return cropped.resize(
        (width, height),
        Image.Resampling.BILINEAR
    )


def random_transform(image):
    transforms = [
        lambda img: jpeg_compress(
            img, random.choice([90, 70, 50])
        ),
        lambda img: gaussian_blur(
            img, random.choice([0.5, 1.0])
        ),
        lambda img: resize_roundtrip(
            img, random.choice([0.75, 0.5])
        ),
        lambda img: gaussian_noise(
            img, random.choice([0.02, 0.05])
        ),
        lambda img: color_jitter(img),
        lambda img: center_crop(img, 0.8),
    ]

    # Keep 30% of training samples clean
    if random.random() < 0.30:
        return image

    # Apply one random robustness transformation
    transform = random.choice(transforms)

    return transform(image)