"""
Train-time / test-time robustness transforms.

Implements the exact transform families and parameter grid from the
challenge brief (section 5.2):

    JPEG Compression   quality in {90, 70, 50, 30}
    Gaussian Blur      sigma in {0.5, 1.0, 2.0}
    Resize             scale 0.5x or 0.25x, then upscale back
    Gaussian Noise     sigma in {0.02, 0.05, 0.10}
    Color Jitter       brightness/contrast/saturation +/- 20%
    Center Crop        crop to 80%, then resize back

Used two ways:
  - the individual functions, called directly with a fixed severity:
    for generating the robustness test grid (clean vs. each
    transform x severity) used in evaluate.py.
  - ALL_CONDITIONS / apply_condition() / sample_condition_names(): a
    named registry over those same functions, PLUS a handful of
    compound (stacked) conditions -- the single source of truth both
    scripts/build_eval_grid.py (the eval grid, applies every named
    condition) and cache_embeddings.py's V2 augmented-variant cache
    (samples a few named conditions per training image) draw from, so
    a condition name can't quietly mean two different things in the
    two places that use it.

All functions take and return a PIL.Image in RGB mode, so they compose
with anything else in the pipeline without extra conversions.
"""

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
COLOR_JITTER_RANGE = 0.2  # +/- 20% on each of brightness/contrast/saturation
CROP_FRACTION = 0.8


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode through JPEG at `quality` and reload -- simulates a social-media re-encode."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur with standard deviation `sigma` -- simulates out-of-focus capture."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_roundtrip(image: Image.Image, scale: float) -> Image.Image:
    """Downscale by `scale` then upscale back to the original size -- simulates thumbnail generation."""
    w, h = image.size
    small = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    """Additive Gaussian noise with std `sigma` (0-1 scale) -- simulates low-light sensor noise."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noisy = arr + np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8), mode="RGB")


def color_jitter(image: Image.Image, jitter_range: float = COLOR_JITTER_RANGE, return_factors: bool = False):
    """Randomly scale brightness, contrast, and saturation each by +/- `jitter_range` -- simulates filter apps / auto-enhance.

    Set return_factors=True to also get back the (brightness, contrast,
    saturation) factors that were sampled -- useful when a call looks
    like it did nothing: a factor near 1.0 is a valid, if visually
    subtle, sample from the allowed range, not a bug.
    """
    out = image.convert("RGB")
    factors = []
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-jitter_range, jitter_range)
        out = enhancer_cls(out).enhance(factor)
        factors.append(factor)
    if return_factors:
        return out, tuple(factors)
    return out


def center_crop(image: Image.Image, crop_fraction: float = CROP_FRACTION) -> Image.Image:
    """Crop the center `crop_fraction` of the image, then resize back -- simulates profile-picture cropping/framing."""
    w, h = image.size
    cw, ch = round(w * crop_fraction), round(h * crop_fraction)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = image.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BICUBIC)


# ---------------------------------------------------------------------------
# Named condition registry (V2)
# ---------------------------------------------------------------------------
#
# SINGLE_CONDITIONS: one entry per brief-grid transform x severity, named
# exactly the way scripts/build_eval_grid.py already named them (e.g.
# "jpeg_q30", "blur_sigma2.0") so this registry is a drop-in replacement
# for that script's previously-hand-rolled condition list, not a new
# naming scheme to reconcile against the eval results already on disk.
SINGLE_CONDITIONS = {"clean": lambda im: im}
for _q in JPEG_QUALITIES:
    SINGLE_CONDITIONS[f"jpeg_q{_q}"] = lambda im, q=_q: jpeg_compress(im, q)
for _s in BLUR_SIGMAS:
    SINGLE_CONDITIONS[f"blur_sigma{_s}"] = lambda im, s=_s: gaussian_blur(im, s)
for _scale in RESIZE_SCALES:
    SINGLE_CONDITIONS[f"resize_{_scale}"] = lambda im, scale=_scale: resize_roundtrip(im, scale)
for _s in NOISE_SIGMAS:
    SINGLE_CONDITIONS[f"noise_sigma{_s}"] = lambda im, s=_s: gaussian_noise(im, s)
SINGLE_CONDITIONS["color_jitter"] = lambda im: color_jitter(im, COLOR_JITTER_RANGE)
SINGLE_CONDITIONS["center_crop"] = lambda im: center_crop(im, CROP_FRACTION)
del _q, _s, _scale

# COMPOUND_CONDITIONS: 2-3 transforms chained in sequence, per the
# architecture doc's V2 scope note ("real-world images are rarely hit
# with exactly one degradation"). Four representative real-world
# scenarios rather than every possible combination (19*18*17*... is not
# a hackathon-scale grid) -- each one chosen to be a plausible single
# story, not an arbitrary transform triple:
COMPOUND_CONDITIONS = {
    # An ordinary phone photo, slightly soft, thumbnailed, then reposted
    # through a standard social-media re-encode.
    "stack_mild": lambda im: jpeg_compress(resize_roundtrip(gaussian_blur(im, 0.5), 0.5), 70),
    # A noisier photo (older sensor / low light) run through a filter
    # app, then reposted.
    "stack_moderate": lambda im: jpeg_compress(color_jitter(gaussian_noise(im, 0.05), COLOR_JITTER_RANGE), 50),
    # The architecture doc's own named worst case, made concrete rather
    # than hypothetical: "blur sigma=2.0, JPEG q=30 stacked with heavy
    # resize" (see Known trade-offs) is now literally this condition.
    "stack_severe": lambda im: jpeg_compress(resize_roundtrip(gaussian_blur(im, 2.0), 0.25), 30),
    # A cropped/reframed repost (e.g. a profile-picture crop) that's
    # also been softened and recompressed.
    "stack_crop_repost": lambda im: jpeg_compress(gaussian_blur(center_crop(im, CROP_FRACTION), 1.0), 50),
}

ALL_CONDITIONS = {**SINGLE_CONDITIONS, **COMPOUND_CONDITIONS}


def condition_names() -> list:
    """All condition names (single-transform + compound), in a stable
    order -- what scripts/build_eval_grid.py iterates over to build the
    full robustness grid."""
    return list(ALL_CONDITIONS.keys())


def apply_condition(image: Image.Image, name: str) -> Image.Image:
    """Applies the named condition to `image`. The single source of
    truth for what a condition name means -- see module docstring."""
    return ALL_CONDITIONS[name](image)


def _weighted_sample_without_replacement(items, weights, k, rng: random.Random):
    """k distinct items, without replacement, drawn with the given
    relative weights. Stdlib `random` has no built-in for this
    (random.choices is with-replacement; random.sample is unweighted),
    and pulling in numpy here would cost this module its "plain
    Python/PIL, no heavy deps" property for a one-off need -- so a
    short manual weighted-removal loop it is."""
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


def sample_condition_names(k: int, rng: random.Random, weight_crop: float = 2.0) -> list:
    """
    k distinct condition names (never "clean" -- callers cache the
    clean embedding separately), for cache_embeddings.py's V2 fixed
    per-image augmented-variant cache.

    "center_crop" is weighted `weight_crop`x over every other
    condition, mirroring the architecture doc's §01 finding (SAFE, KDD
    2025): crop forces a smaller field of view without flattening the
    fine texture the frequency stream depends on the way resize-
    downsample does, so it's preferred when there's a choice -- at
    TRAINING time specifically. scripts/build_eval_grid.py's eval grid
    deliberately does NOT use this sampler; it applies every condition
    equally, since the eval grid isn't trying to teach the model
    anything, only measure it.

    `rng` is passed in (rather than using the module-level `random`
    state) so a caller can seed it once and advance it across many
    rows for a reproducible, stable assignment across reruns -- the
    "fixed set of augmented variants" the architecture doc's caching
    design calls for, not fresh random augmentation every run.
    """
    names = [n for n in ALL_CONDITIONS if n != "clean"]
    weights = [weight_crop if n == "center_crop" else 1.0 for n in names]
    return _weighted_sample_without_replacement(names, weights, k, rng)
