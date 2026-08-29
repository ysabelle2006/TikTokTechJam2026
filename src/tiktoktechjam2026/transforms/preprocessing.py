"""
Per-stream input preparation.

The two model streams see different things:
  - spatial stream:    resized + CLIP-normalized RGB image
  - frequency stream:  a grayscale forensic map -- either an SRM-style
                       high-pass residual (spatial domain) or an FFT
                       log-magnitude spectrum -- that exposes generator
                       artifacts the spatial stream is not looking for

Both are computed from the *already-transformed* pixels: evaluate.py applies
a robustness transform to the raw image first, then calls these. CLIP
normalization and the frequency residual are always the last step, never
something a transform is applied on top of.

`residual_energy` is the scalar reliability signal from the architecture
doc: roughly how much high-frequency energy survived, so the fusion head
has an explicit hint about when to discount the frequency stream (e.g.
after heavy blur it collapses toward zero).
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from tiktoktechjam2026 import config

IMAGE_SIZE = config.IMAGE_SIZE

# The pretrained CLIP backbone was trained with these exact normalization
# numbers, so we must reuse them or the embeddings come out wrong.
CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_spatial_pipeline = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.ToTensor(),                      # -> 3xHxW float in 0..1
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])


def prepare_spatial_input(image):
    """PIL image -> tensor [3, 224, 224] for the CLIP backbone."""
    return _spatial_pipeline(image.convert("RGB"))


# --------------------------------------------------------------------------
# Frequency stream inputs
# --------------------------------------------------------------------------

# Classic SRM / forensic high-pass kernels (normalized). A small bank of
# complementary high-pass filters is the standard spatial-domain way to
# surface resampling / upsampling / denoising residuals.
_SRM_KERNELS = [
    # first-order horizontal+vertical difference
    np.array([[0, 0, 0],
              [0, -1, 1],
              [0, 0, 0]], dtype=np.float32),
    # second-order 3x3 (Laplacian-like)
    np.array([[-1, 2, -1],
              [2, -4, 2],
              [-1, 2, -1]], dtype=np.float32) / 4.0,
    # third-order 5x5 SRM ("KB") kernel
    np.array([[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0,
]
assert len(_SRM_KERNELS) >= config.FREQUENCY_SRM_CHANNELS


def _grayscale_224(image):
    """PIL image -> float32 [224, 224] grayscale in [0, 1]."""
    gray = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(gray, dtype=np.float32) / 255.0


def high_pass_residual(image):
    """
    Simple blur-residual (original - Gaussian(sigma=1)) as float32 [224, 224].

    This is the map `residual_energy` measures -- kept separate from the
    CNN input so the reliability scalar means the same thing regardless of
    which FREQUENCY_MODE the CNN is using.
    """
    gray = _grayscale_224(image)
    blurred = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=1.0)
    return gray - blurred


def _srm_residual(image):
    """Grayscale -> [FREQUENCY_SRM_CHANNELS, 224, 224] stack of SRM residuals."""
    gray = _grayscale_224(image)
    maps = [
        cv2.filter2D(gray, ddepth=-1, kernel=k)
        for k in _SRM_KERNELS[:config.FREQUENCY_SRM_CHANNELS]
    ]
    return np.stack(maps, axis=0).astype(np.float32)


def _fft_logmag(image):
    """Grayscale -> [1, 224, 224] FFT log-magnitude spectrum, normalized to [0, 1]."""
    gray = _grayscale_224(image)
    spectrum = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    mag = np.log1p(np.abs(spectrum)).astype(np.float32)
    lo, hi = float(mag.min()), float(mag.max())
    if hi > lo:
        mag = (mag - lo) / (hi - lo)
    return mag[None, :, :]


def prepare_frequency_input(image, mode=None):
    """
    PIL image -> CNN input tensor for the frequency stream.

    mode "srm": [FREQUENCY_SRM_CHANNELS, 224, 224] SRM high-pass residuals
    mode "fft": [1, 224, 224] FFT log-magnitude spectrum
    """
    mode = mode or config.FREQUENCY_MODE
    image = image.convert("RGB")
    if mode == "srm":
        arr = _srm_residual(image)
    elif mode == "fft":
        arr = _fft_logmag(image)
    else:
        raise ValueError(f"unknown FREQUENCY_MODE: {mode!r}")
    return torch.from_numpy(arr)


def frequency_input_channels(mode=None):
    """Number of input channels the frequency CNN should expect for `mode`."""
    mode = mode or config.FREQUENCY_MODE
    return config.FREQUENCY_SRM_CHANNELS if mode == "srm" else 1


def residual_energy(residual_map):
    """
    Reliability signal for the fusion head: the standard deviation of the
    high-pass residual. Low energy (e.g. after heavy blur or strong JPEG)
    tells the fusion head the frequency stream is currently unreliable.

    Accepts either a residual map (tensor / ndarray, as from
    `high_pass_residual`) or a PIL image (in which case the blur residual is
    computed here). Returns a single float.
    """
    if isinstance(residual_map, Image.Image):
        residual_map = high_pass_residual(residual_map)
    if isinstance(residual_map, torch.Tensor):
        return float(residual_map.std())
    return float(np.asarray(residual_map).std())
