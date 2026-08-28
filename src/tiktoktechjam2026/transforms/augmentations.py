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

def jpeg_compress(image, quality: int):
    buffer = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffer, format = "JPEG", quality = quality)
    buffer.seek(0)
    return Image.open(buffer)

def gaussian_blur(image, sigma: float):
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))
    

def resize_roundtrip(image, scale: float):
    w,h = image.size
    small_w = max(1, round(w*scale))
    small_h = max(1, round(h*scale))
    downscale = image.resize((small_w, small_h), Image.BILINEAR)
    return downscale.resize((w,h),Image.BILINEAR)


def gaussian_noise(image, sigma: float,RNG = None):
    if RNG is None:
        RNG = np.random.default_rng()
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noise = RNG.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))


def color_jitter(image, brightness: float, contrast: float, saturation: float):
    brightness_enhancer = ImageEnhance.Brightness(image)
    bright_image = brightness_enhancer.enhance(brightness)
    contrast_enhancer = ImageEnhance.Contrast(bright_image)
    contrast_image = contrast_enhancer.enhance(contrast)
    saturation_enhancer = ImageEnhance.Color(contrast_image)
    saturation_image = saturation_enhancer.enhance(saturation)
    return saturation_image


def center_crop(image, crop_fraction: float = 0.8):
    w,h = image.size
    crop_w = w * crop_fraction
    crop_h = h * crop_fraction
    left = round((w - crop_w) / 2)
    top = round((h - crop_h) / 2)
    right = round((w + crop_w) / 2)
    bottom = round((h + crop_h) / 2)
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image.resize((w,h),Image.BILINEAR)


def random_transform(image, rng = None):
    """Apply a random subset of the above, at a random severity. TODO."""
    if rng is None:
        rng = np.random.default_rng()
    image = image.convert("RGB")
    ops = [
    lambda im: jpeg_compress(im, int(rng.choice([90, 70, 50, 30]))),
    lambda im: gaussian_blur(im, float(rng.choice([0.5, 1.0, 2.0]))),
    lambda im: resize_roundtrip(im, float(rng.choice([0.5, 0.25]))),
    lambda im: gaussian_noise(im, float(rng.choice([0.02, 0.05, 0.10])), rng),
    lambda im: color_jitter(im, *rng.uniform(0.8, 1.2, size=3)),
    lambda im: center_crop(im, 0.8),
    ]
    k = int(rng.integers(0, 3))
    if k == 0:
        return image

    chosen = rng.choice(len(ops), size=k, replace=False)
    out = image
    for i in chosen:
        out = ops[int(i)](out)
    return out