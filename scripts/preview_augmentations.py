"""
Visual + numeric smoke test for transforms/augmentations.py -- no
dataset needed.

Generates one synthetic test image, applies every transform x severity
from the brief's grid, saves previews, and -- because eyeballing small
thumbnails isn't a reliable enough test on its own (resize and blur
can look deceptively similar, and a random color-jitter draw can land
close to identity) -- also prints two numbers per output:

  - variance of the Laplacian: a standard blur metric (lower = less
    high-frequency detail survived). Comparable across ANY transform,
    which is exactly what we want to check whether resize and blur are
    doing the same thing to the image or merely looking similar in a
    150px thumbnail.
  - mean absolute pixel difference from the clean image: a blunt "how
    much did this change" number, useful for confirming a color-jitter
    call with subtle sampled factors actually changed something.

Run with:  uv run python scripts/preview_augmentations.py
"""

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transforms.augmentations import (
    BLUR_SIGMAS,
    COLOR_JITTER_RANGE,
    CROP_FRACTION,
    JPEG_QUALITIES,
    NOISE_SIGMAS,
    RESIZE_SCALES,
    center_crop,
    color_jitter,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
    resize_roundtrip,
)

OUT_DIR = Path("outputs/augmentation_preview")


def make_test_image(size=224):
    """A synthetic image with edges, color, and texture -- so distortions are visible."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        arr[y, :, 0] = int(255 * y / size)
        arr[y, :, 2] = 255 - int(255 * y / size)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse((size * 0.2, size * 0.2, size * 0.8, size * 0.8), outline=(0, 255, 0), width=6)
    draw.rectangle((size * 0.05, size * 0.05, size * 0.35, size * 0.35), outline=(255, 255, 0), width=4)
    for i in range(0, size, 16):
        draw.line((i, 0, i, size // 6), fill=(255, 255, 255), width=1)
    draw.text((size * 0.3, size * 0.45), "TEST", fill=(255, 255, 255))
    return img


def variance_of_laplacian(image: Image.Image) -> float:
    """Standard blur metric: variance of a Laplacian-filtered grayscale image.
    Lower = flatter / less high-frequency detail survived, whatever the cause."""
    gray = image.convert("L")
    laplacian = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1))
    return float(np.asarray(laplacian, dtype=np.float64).var())


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute per-pixel difference between two same-size RGB images (0-255 scale)."""
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float64)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.float64)
    return float(np.abs(arr_a - arr_b).mean())


def main():
    random.seed(0)
    np.random.seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = make_test_image()
    clean.save(OUT_DIR / "00_clean.png")
    clean_vol = variance_of_laplacian(clean)

    rows = [("clean", clean, clean_vol, 0.0)]

    def record(name, img):
        img.save(OUT_DIR / f"{name}.png")
        rows.append((name, img, variance_of_laplacian(img), mean_abs_diff(clean, img)))

    for q in JPEG_QUALITIES:
        record(f"jpeg_q{q}", jpeg_compress(clean, q))
    for s in BLUR_SIGMAS:
        record(f"blur_sigma{s}", gaussian_blur(clean, s))
    for scale in RESIZE_SCALES:
        record(f"resize_{scale}", resize_roundtrip(clean, scale))
    for s in NOISE_SIGMAS:
        record(f"noise_sigma{s}", gaussian_noise(clean, s))
    for name, cls, factor in [
        ("jitter_brightness_up", ImageEnhance.Brightness, 1 + COLOR_JITTER_RANGE),
        ("jitter_brightness_down", ImageEnhance.Brightness, 1 - COLOR_JITTER_RANGE),
        ("jitter_contrast_up", ImageEnhance.Contrast, 1 + COLOR_JITTER_RANGE),
        ("jitter_contrast_down", ImageEnhance.Contrast, 1 - COLOR_JITTER_RANGE),
        ("jitter_saturation_up", ImageEnhance.Color, 1 + COLOR_JITTER_RANGE),
        ("jitter_saturation_down", ImageEnhance.Color, 1 - COLOR_JITTER_RANGE),
    ]:
        record(name, cls(clean).enhance(factor))
    jittered, factors = color_jitter(clean, COLOR_JITTER_RANGE, return_factors=True)
    record("color_jitter_random", jittered)
    record("center_crop", center_crop(clean, CROP_FRACTION))

    print(f"{'name':<24}{'var(Laplacian)':>16}{'mean|diff| vs clean':>22}")
    for name, _img, vol, diff in rows:
        print(f"{name:<24}{vol:>16.1f}{diff:>22.2f}")
    print(f"\ncolor_jitter_random sampled factors (brightness, contrast, saturation): "
          f"{tuple(round(f, 3) for f in factors)}")
    print(f"\nWrote {len(list(OUT_DIR.glob('*.png')))} preview images to {OUT_DIR}/")


if __name__ == "__main__":
    main()
