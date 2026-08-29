"""
Frequency-stream input preparation.

The spatial stream's preprocessing is handled directly by
SpatialStream.prepare() (models/spatial_stream.py), which uses CLIP's
own transform rather than a hand-rolled one -- so this module's scope
is just what the frequency stream needs:

  - grayscale, high-pass-filtered residual (or FFT magnitude map) that
    exposes generator artifacts
  - the residual-energy scalar from the architecture doc: roughly, how
    much high-frequency energy survived, used to help the fusion head
    know when to discount the frequency stream

Deliberately torch-free -- everything here is plain numpy/scipy/PIL,
which means (unlike models/frequency_stream.py) it's testable in any
sandbox without torch installed, and it's also what train.py's V1
dataset calls directly in DataLoader worker processes (see that
module's _FrequencyTrainDataset docstring for why it goes around
FrequencyStream rather than through it there). See
scripts/preview_frequency_input.py.

models.frequency_stream.FrequencyStream.prepare() is what converts
this module's numpy output into a torch tensor (plus the residual-
energy scalar) for its CNN, the same way SpatialStream.prepare() wraps
CLIP's preprocessing.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

from config import FREQUENCY_INPUT_SIZE, FREQUENCY_MODE

# SRM "KV" high-pass kernel (the Ker/Boehme steganalysis filter), a
# fixed, untrained high-pass filter widely reused in image-forensics
# and GAN/diffusion-artifact detection work (e.g. Zhou et al., "Learning
# Rich Features for Image Manipulation Detection", CVPR 2018) precisely
# because it suppresses semantic image content and amplifies the local
# pixel-correlation residue that generator decoders tend to leave
# behind. Normalized by /12 so residual magnitudes stay in a sane range
# without needing extra per-image rescaling downstream.
_SRM_KV_KERNEL = np.array(
    [
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ],
    dtype=np.float32,
) / 12.0


def _to_grayscale_array(image: Image.Image) -> np.ndarray:
    """PIL.Image (RGB) -> float32 grayscale array, values in [0, 1], at
    the image's native resolution."""
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _srm_residual(gray: np.ndarray) -> np.ndarray:
    """5x5 high-pass filter -> residual map, same size as `gray`.

    mode="mirror" avoids the dark border a zero-pad would introduce at
    the edges -- we want the residual to reflect the image's own
    texture near the border, not an artifact of how convolution
    handled the boundary.
    """
    return convolve(gray, _SRM_KV_KERNEL, mode="mirror")


def _fft_magnitude(gray: np.ndarray) -> np.ndarray:
    """Log-scaled magnitude of the 2D FFT, DC component centered.

    Generator upsampling (transposed convolutions / pixel-shuffle)
    tends to leave periodic checkerboard patterns that show up as
    symmetric bright peaks away from the DC in this spectrum -- the
    classic frequency-domain GAN-fingerprint signal (Zhang et al.,
    "Detecting and Simulating Artifacts in GAN Fake Images", 2019;
    Frank et al., "Leveraging Frequency Analysis for Deep Fake Image
    Recognition", 2020).

    log1p rather than log so this stays finite even where the raw
    magnitude is exactly 0 (a real possibility for a very flat patch).
    """
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    return np.log1p(np.abs(spectrum)).astype(np.float32)


def _resize_map(arr: np.ndarray, size: int) -> np.ndarray:
    """Bicubic-resize a single-channel float32 map to `size` x `size`.

    Round-trips through PIL's "F" (32-bit float) image mode rather than
    the uint8 mode used elsewhere in this codebase, specifically so
    this does NOT clip or rescale the values. An SRM residual is
    centered near zero with meaningful negative values, and
    residual_energy() below needs the actual, un-renormalized
    magnitudes to stay comparable across different images (e.g. clean
    vs. blurred) and across the two modes. uint8 mode would silently
    clip everything below 0 and force a per-image min/max stretch,
    which would erase exactly the amplitude differences
    residual_energy() is trying to measure.
    """
    as_img = Image.fromarray(arr.astype(np.float32), mode="F")
    resized = as_img.resize((size, size), Image.BICUBIC)
    # np.asarray(resized, dtype=np.float32) would wrap PIL's own buffer
    # without copying (since the dtype already matches "F" mode), which
    # PIL marks read-only -- torch.from_numpy() on that later warns
    # ("non-writable tensor... undefined behavior") on every single
    # image. .copy() forces an actual owned, writable array here once,
    # rather than that warning firing on every training step.
    return np.asarray(resized, dtype=np.float32).copy()


def prepare_frequency_input(image: Image.Image, mode: str = None, size: int = None) -> np.ndarray:
    """
    PIL.Image (RGB) -> a single-channel float32 map, shape (size, size).
    Values are intentionally left un-renormalized (see _resize_map) so
    residual_energy() can measure real amplitude, not a per-image
    rescaled version of it. models.frequency_stream.FrequencyStream
    handles converting this to a tensor and feeding it through a
    BatchNorm2d input layer, which normalizes at the batch level
    instead -- see that module for why that split matters.

    mode: "srm" or "fft" (defaults to config.FREQUENCY_MODE). The two
    differ in WHEN they resize relative to the residual extraction,
    because the ordering matters differently for each:

      - "srm": the high-pass filter runs at the image's native
        resolution first, then the residual map itself is resized to
        `size`. Filtering before resizing preserves the fine
        pixel-level texture the filter is looking for; resizing a
        residual (rather than the raw image) still smooths it
        somewhat, which is a known limitation for very small source
        images (e.g. CIFAKE's 32x32) being upsampled to `size`.
      - "fft": the image is resized to `size` BEFORE the FFT, since the
        spectrum's bin-to-frequency mapping depends on the image's
        pixel dimensions -- resizing first makes spectra comparable
        across source images of different native sizes, which "srm"'s
        per-pixel residual doesn't need.

    size: output side length (defaults to config.FREQUENCY_INPUT_SIZE).
    """
    mode = mode or FREQUENCY_MODE
    size = size or FREQUENCY_INPUT_SIZE
    gray = _to_grayscale_array(image)

    if mode == "srm":
        residual = _srm_residual(gray)
        return _resize_map(residual, size)
    elif mode == "fft":
        resized_gray = _resize_map(gray, size)
        return _fft_magnitude(resized_gray)
    else:
        raise ValueError(f"unknown frequency mode {mode!r}, expected 'srm' or 'fft'")


def residual_energy(freq_map: np.ndarray) -> float:
    """
    Scalar summary of how much high-frequency energy survived into
    `freq_map` (the output of prepare_frequency_input, BEFORE any
    network sees it -- see that function's docstring on why it isn't
    renormalized). Meant for the fusion head (per the architecture doc)
    to discount the frequency stream's vote when this is low -- e.g.
    after a strong Gaussian blur has wiped out the fine texture the
    frequency stream depends on, the fusion head should lean more on
    the spatial stream instead.

    Root-mean-square rather than mean-absolute deviation: squaring
    before averaging weights a few strong residual spikes (the kind of
    signal actually worth detecting) more than many small ones, and
    matches "energy" in the signal-processing sense (mean squared
    amplitude) rather than plain average deviation.
    """
    return float(np.sqrt(np.mean(np.square(freq_map))))
