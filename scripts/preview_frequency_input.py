"""
Smoke test for transforms/preprocessing.py -- no dataset, no torch
needed (unlike preview_spatial_stream.py / preview_frequency_stream.py,
this module is plain numpy/scipy/PIL, so this script actually runs in
any sandbox -- and, unlike the two model-level smoke tests, never
depended on a torch install to be verified in the first place).

What it checks, and why each check matters:
  - prepare_frequency_input() returns the expected shape for both
    "srm" and "fft" modes (catches an obvious wiring bug early).
  - a synthetic checkerboard region (standing in for the periodic
    upsampling artifact real generators leave) has a HIGHER SRM
    residual_energy than a smooth gradient region of the same size --
    a real pass/fail signal that the high-pass filter is actually
    picking up high-frequency structure, not just returning noise.
  - Gaussian-blurring an image LOWERS its residual_energy -- this is
    the exact property the fusion head depends on (discount the
    frequency stream once blur has washed out the fine texture it
    needs), so if blur did NOT reduce energy here, that fusion-head
    assumption would be built on sand.

Run with:  python scripts/preview_frequency_input.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import FREQUENCY_INPUT_SIZE
from transforms.augmentations import gaussian_blur
from transforms.preprocessing import prepare_frequency_input, residual_energy

OUT_DIR = Path("outputs/frequency_preview")


def make_checkerboard(size=224, cell=4):
    """Tight checkerboard -- a crude stand-in for the periodic grid
    artifact strided transposed-convolution upsampling tends to leave;
    real generator fingerprints are subtler than this, but this is
    deliberately an easy, unambiguous positive case for the high-pass
    filter to catch."""
    arr = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if (x // cell + y // cell) % 2 == 0:
                arr[y, x] = 255
    return Image.fromarray(arr, mode="L").convert("RGB")


def make_smooth_gradient(size=224):
    """Smooth gradient, no high-frequency content -- the negative case."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        arr[y, :, :] = int(255 * y / size)
    return Image.fromarray(arr, mode="RGB")


def make_test_photo(size=224):
    """A synthetic "photo-like" image with edges and shapes -- realistic
    enough to sensibly run blur/JPEG on, unlike the checkerboard/
    gradient extremes above."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        arr[y, :, 0] = int(255 * y / size)
        arr[y, :, 2] = 255 - int(255 * y / size)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse((size * 0.2, size * 0.2, size * 0.8, size * 0.8), outline=(0, 255, 0), width=3)
    for i in range(0, size, 8):
        draw.line((i, 0, i, size // 4), fill=(255, 255, 255), width=1)
    return img


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"FREQUENCY_INPUT_SIZE = {FREQUENCY_INPUT_SIZE}\n")

    photo = make_test_photo()
    for mode in ("srm", "fft"):
        freq_map = prepare_frequency_input(photo, mode=mode)
        print(f"mode={mode:<4} output shape: {freq_map.shape} "
              f"(expect ({FREQUENCY_INPUT_SIZE}, {FREQUENCY_INPUT_SIZE}))")
        assert freq_map.shape == (FREQUENCY_INPUT_SIZE, FREQUENCY_INPUT_SIZE), "shape mismatch!"
        assert np.isfinite(freq_map).all(), f"non-finite values in mode={mode} output!"

    checker = make_checkerboard()
    smooth = make_smooth_gradient()
    energy_checker = residual_energy(prepare_frequency_input(checker, mode="srm"))
    energy_smooth = residual_energy(prepare_frequency_input(smooth, mode="srm"))
    print(f"\nSRM residual_energy, checkerboard (high-freq):  {energy_checker:.4f}")
    print(f"SRM residual_energy, smooth gradient (low-freq): {energy_smooth:.4f}")
    if energy_checker > energy_smooth:
        print("PASS: checkerboard scores higher energy than a smooth gradient, as expected.")
    else:
        print("UNEXPECTED: smooth gradient scored >= checkerboard -- the high-pass "
              "filter isn't behaving as expected, don't trust this yet.")

    energy_clean = residual_energy(prepare_frequency_input(photo, mode="srm"))
    blurred = gaussian_blur(photo, sigma=2.0)
    energy_blurred = residual_energy(prepare_frequency_input(blurred, mode="srm"))
    print(f"\nSRM residual_energy, clean photo:          {energy_clean:.4f}")
    print(f"SRM residual_energy, same photo blurred:   {energy_blurred:.4f}")
    if energy_blurred < energy_clean:
        print("PASS: blurring reduced residual energy, as the fusion head's "
              "discounting logic assumes.")
    else:
        print("UNEXPECTED: blur did not reduce residual energy -- the fusion head's "
              "planned 'discount frequency stream after blur' logic would be wrong.")

    clean_map = prepare_frequency_input(photo, mode="srm")
    Image.fromarray(
        ((clean_map - clean_map.min()) / np.ptp(clean_map) * 255).astype(np.uint8),
        mode="L",
    ).save(OUT_DIR / "srm_residual_photo.png")
    photo.save(OUT_DIR / "00_clean_photo.png")
    checker.save(OUT_DIR / "checkerboard.png")
    print(f"\nWrote preview images to {OUT_DIR}/")


if __name__ == "__main__":
    main()
