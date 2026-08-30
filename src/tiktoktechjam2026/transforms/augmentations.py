"""
Train-time / test-time robustness transforms.

Implements the exact transform families and parameter grid from the
challenge brief:

    JPEG Compression   quality in {90, 70, 50, 30}
    Gaussian Blur      sigma in {0.5, 1.0, 2.0}
    Resize             scale 0.5x or 0.25x, then upscale back
    Gaussian Noise     sigma in {0.02, 0.05, 0.10}
    Color Jitter       brightness/contrast/saturation +/- 20%
    Center Crop        crop to 80%, then resize back

Also defines four stacked conditions for V2-style robustness evaluation.
"""

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
COLOR_JITTER_RANGE = 0.2
CROP_FRACTION = 0.8


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_roundtrip(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size

    small = image.resize(
        (
            max(1, round(w * scale)),
            max(1, round(h * scale)),
        ),
        Image.BICUBIC,
    )

    return small.resize((w, h), Image.BICUBIC)


def gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    ) / 255.0

    noisy = arr + np.random.normal(
        0.0,
        sigma,
        arr.shape,
    ).astype(np.float32)

    noisy = np.clip(noisy, 0.0, 1.0)

    return Image.fromarray(
        (noisy * 255.0).astype(np.uint8),
        mode="RGB",
    )


def color_jitter(
    image: Image.Image,
    jitter_range: float = COLOR_JITTER_RANGE,
    return_factors: bool = False,
):
    out = image.convert("RGB")
    factors = []

    for enhancer_cls in (
        ImageEnhance.Brightness,
        ImageEnhance.Contrast,
        ImageEnhance.Color,
    ):
        factor = 1.0 + random.uniform(
            -jitter_range,
            jitter_range,
        )

        out = enhancer_cls(out).enhance(factor)
        factors.append(factor)

    if return_factors:
        return out, tuple(factors)

    return out


def center_crop(
    image: Image.Image,
    crop_fraction: float = CROP_FRACTION,
) -> Image.Image:
    w, h = image.size

    cw = round(w * crop_fraction)
    ch = round(h * crop_fraction)

    left = (w - cw) // 2
    top = (h - ch) // 2

    cropped = image.crop(
        (
            left,
            top,
            left + cw,
            top + ch,
        )
    )

    return cropped.resize((w, h), Image.BICUBIC)


# ---------------------------------------------------------
# Named condition registry
# ---------------------------------------------------------

SINGLE_CONDITIONS = {
    "clean": lambda im: im,
}

for _q in JPEG_QUALITIES:
    SINGLE_CONDITIONS[f"jpeg_q{_q}"] = (
        lambda im, q=_q: jpeg_compress(im, q)
    )

for _s in BLUR_SIGMAS:
    SINGLE_CONDITIONS[f"blur_sigma{_s}"] = (
        lambda im, s=_s: gaussian_blur(im, s)
    )

for _scale in RESIZE_SCALES:
    SINGLE_CONDITIONS[f"resize_{_scale}"] = (
        lambda im, scale=_scale: resize_roundtrip(im, scale)
    )

for _s in NOISE_SIGMAS:
    SINGLE_CONDITIONS[f"noise_sigma{_s}"] = (
        lambda im, s=_s: gaussian_noise(im, s)
    )

SINGLE_CONDITIONS["color_jitter"] = (
    lambda im: color_jitter(
        im,
        COLOR_JITTER_RANGE,
    )
)

SINGLE_CONDITIONS["center_crop"] = (
    lambda im: center_crop(
        im,
        CROP_FRACTION,
    )
)

del _q, _s, _scale


COMPOUND_CONDITIONS = {
    "stack_mild": lambda im: jpeg_compress(
        resize_roundtrip(
            gaussian_blur(im, 0.5),
            0.5,
        ),
        70,
    ),

    "stack_moderate": lambda im: jpeg_compress(
        color_jitter(
            gaussian_noise(im, 0.05),
            COLOR_JITTER_RANGE,
        ),
        50,
    ),

    "stack_severe": lambda im: jpeg_compress(
        resize_roundtrip(
            gaussian_blur(im, 2.0),
            0.25,
        ),
        30,
    ),

    "stack_crop_repost": lambda im: jpeg_compress(
        gaussian_blur(
            center_crop(
                im,
                CROP_FRACTION,
            ),
            1.0,
        ),
        50,
    ),
}


ALL_CONDITIONS = {
    **SINGLE_CONDITIONS,
    **COMPOUND_CONDITIONS,
}


def condition_names() -> list:
    return list(ALL_CONDITIONS.keys())


def apply_condition(
    image: Image.Image,
    name: str,
) -> Image.Image:
    return ALL_CONDITIONS[name](image)


def _weighted_sample_without_replacement(
    items,
    weights,
    k,
    rng: random.Random,
):
    items = list(items)
    weights = list(weights)
    chosen = []

    for _ in range(min(k, len(items))):
        total = sum(weights)
        r = rng.random() * total
        upto = 0.0

        for i, w in enumerate(weights):
            upto += w

            if upto >= r:
                chosen.append(items.pop(i))
                weights.pop(i)
                break

    return chosen


def sample_condition_names(
    k: int,
    rng: random.Random,
    weight_crop: float = 2.0,
) -> list:
    names = [
        n
        for n in ALL_CONDITIONS
        if n != "clean"
    ]

    weights = [
        weight_crop if n == "center_crop" else 1.0
        for n in names
    ]

    return _weighted_sample_without_replacement(
        names,
        weights,
        k,
        rng,
    )