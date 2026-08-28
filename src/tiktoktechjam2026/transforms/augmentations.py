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


def jpeg_compress(image, quality: int):
    raise NotImplementedError


def gaussian_blur(image, sigma: float):
    raise NotImplementedError


def resize_roundtrip(image, scale: float):
    raise NotImplementedError


def gaussian_noise(image, sigma: float):
    raise NotImplementedError


def color_jitter(image, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2):
    raise NotImplementedError


def center_crop(image, crop_fraction: float = 0.8):
    raise NotImplementedError


def random_transform(image):
    """Apply a random subset of the above, at a random severity. TODO."""
    raise NotImplementedError
