"""
Robustness transforms from the challenge brief (section 5.2).

    JPEG compression   quality in {90, 70, 50, 30}
    Gaussian blur      sigma in {0.5, 1.0, 2.0}
    Resize             scale 0.5x or 0.25x, then upscale back
    Gaussian noise     sigma in {0.02, 0.05, 0.10}   (sigma is in [0, 1] pixel units)
    Color jitter       exactly ONE of brightness / contrast / saturation, factor in [0.8, 1.2]
    Center crop        crop to 80% of each side, then resize back

Two consumers:

  * evaluate.py -- applies exactly ONE transform at ONE severity to the raw
    image (native resolution, RGB, 8-bit, sRGB) before either stream's
    preprocessing. `apply_condition` is the single entry point; the grid it
    dispatches over is config.EVAL_CONDITIONS.

  * train.py (V2 onwards) -- `random_transform` applies a random subset
    (1..config.AUG_MAX_SIMULTANEOUS, never all six) at random severities, so
    the model trains through the same post-processing it will be tested on.
    V0 and V1 do not use this; they train on clean images only.

All operations run in the 8-bit sRGB pixel domain, before CLIP normalization
and before the frequency residual -- both streams recompute from the
transformed pixels.
"""

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from tiktoktechjam2026 import config

# Severity menus for training-time augmentation (evaluate.py passes explicit
# severities straight from config.EVAL_CONDITIONS instead).
JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
COLOR_JITTER_RANGE = (0.8, 1.2)
CROP_FRACTION = 0.8


# --------------------------------------------------------------------------
# Individual transforms.  Each takes and returns a PIL.Image in RGB.
# --------------------------------------------------------------------------

def jpeg_compress(image, quality):
    """Re-encode as JPEG at `quality` (libjpeg 1-100 scale) and decode back."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    # .copy() detaches the pixels from the BytesIO before it is garbage-collected.
    return Image.open(buffer).convert("RGB").copy()


def gaussian_blur(image, sigma):
    """
    Gaussian blur with standard deviation `sigma` in pixels.

    PIL's GaussianBlur `radius` argument is documented as the kernel standard
    deviation, so radius == sigma here.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize_roundtrip(image, scale):
    """Downscale each side by `scale`, then upscale back to the original size."""
    w, h = image.size
    small_w = max(1, round(w * scale))
    small_h = max(1, round(h * scale))
    small = image.resize((small_w, small_h), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def gaussian_noise(image, sigma, rng=None):
    """
    Add i.i.d. Gaussian noise, per pixel and per channel.

    `sigma` is expressed in [0, 1] pixel units (sigma=0.02 ~= 5.1/255), so the
    noise is added after scaling the image to [0, 1] and the result is clipped
    back to that range.
    """
    if rng is None:
        rng = np.random.default_rng()
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noise = rng.normal(0.0, float(sigma), size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB")


_COLOR_JITTER_ENHANCERS = {
    "brightness": ImageEnhance.Brightness,
    "contrast": ImageEnhance.Contrast,
    "saturation": ImageEnhance.Color,
}


def color_jitter(image, channel, factor):
    """
    Apply a single enhancement (`channel` in {brightness, contrast, saturation})
    by `factor` (1.0 == identity; the brief's +/-20% -> factor in [0.8, 1.2]).
    """
    enhancer_cls = _COLOR_JITTER_ENHANCERS[channel]
    return enhancer_cls(image.convert("RGB")).enhance(float(factor))


def center_crop(image, crop_fraction=CROP_FRACTION):
    """Center-crop to `crop_fraction` of each side, then resize back up."""
    w, h = image.size
    crop_w = w * crop_fraction
    crop_h = h * crop_fraction
    left = round((w - crop_w) / 2)
    top = round((h - crop_h) / 2)
    right = round((w + crop_w) / 2)
    bottom = round((h + crop_h) / 2)
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.BILINEAR)


# --------------------------------------------------------------------------
# Evaluation dispatch: one transform, one fixed severity.
# --------------------------------------------------------------------------

def apply_condition(image, transform_name, param, rng=None):
    """
    Apply a single robustness condition to a raw PIL image.

    `transform_name` / `param` come straight from a config.EVAL_CONDITIONS
    row. `rng` (seeded per-image by the caller) is used only by the
    stochastic conditions -- noise and color jitter -- so that a given
    (image, condition) pair renders identically on every run.
    """
    image = image.convert("RGB")
    if rng is None:
        rng = np.random.default_rng(0)

    if transform_name == "identity":
        return image
    if transform_name == "jpeg":
        return jpeg_compress(image, param)
    if transform_name == "blur":
        return gaussian_blur(image, param)
    if transform_name == "resize":
        return resize_roundtrip(image, param)
    if transform_name == "noise":
        return gaussian_noise(image, param, rng)
    if transform_name == "crop":
        return center_crop(image, param)
    if transform_name == "color_jitter":
        # `param` is the +/- fraction (0.20). One channel, one factor.
        channel = rng.choice(list(_COLOR_JITTER_ENHANCERS))
        factor = float(rng.uniform(1.0 - param, 1.0 + param))
        return color_jitter(image, channel, factor)

    raise ValueError(f"unknown transform: {transform_name!r}")


# --------------------------------------------------------------------------
# Training dispatch (V2+): a random subset at random severities.
# --------------------------------------------------------------------------

def _train_ops(rng):
    """The six transform families as zero-arg closures over a random severity."""
    return [
        lambda im: jpeg_compress(im, int(rng.choice(JPEG_QUALITIES))),
        lambda im: gaussian_blur(im, float(rng.choice(BLUR_SIGMAS))),
        lambda im: resize_roundtrip(im, float(rng.choice(RESIZE_SCALES))),
        lambda im: gaussian_noise(im, float(rng.choice(NOISE_SIGMAS)), rng),
        lambda im: color_jitter(
            im,
            rng.choice(list(_COLOR_JITTER_ENHANCERS)),
            float(rng.uniform(*COLOR_JITTER_RANGE)),
        ),
        lambda im: center_crop(im, CROP_FRACTION),
    ]


def random_transform(image, rng=None):
    """
    Apply a random subset of the six transforms, in random order, each at a
    random severity. The subset size is uniform on
    [1, config.AUG_MAX_SIMULTANEOUS] -- never all six at once, matching the
    brief's "a subset of these transformations" wording.

    Training-time only (V2 onwards). Returns a PIL.Image in RGB.
    """
    if rng is None:
        rng = np.random.default_rng()
    image = image.convert("RGB")

    ops = _train_ops(rng)
    k = int(rng.integers(1, config.AUG_MAX_SIMULTANEOUS + 1))
    chosen = rng.permutation(len(ops))[:k]

    out = image
    for i in chosen:
        out = ops[int(i)](out)
    return out
