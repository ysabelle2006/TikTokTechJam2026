"""
Frequency stream: high-pass/forensic residual + small CNN, trained from
scratch.

Reads for the low-level statistical tell a generator leaves behind --
GAN upsampling checkerboards, diffusion denoising residuals -- which
the spatial stream isn't looking for. This is the fragile signal:
Gaussian blur is a low-pass filter, so it removes exactly what this
stream depends on, and heavy JPEG quantizes it away too. JPEG directly
manipulates frequency-domain coefficients, so it may hit an FFT-based
version of this stream harder than a spatial-domain residual, or the
other way around -- we don't know yet, and shouldn't assume.

Deliberately NOT committing to one artifact-extraction method. Implement
both and run them as an ablation (config.FREQUENCY_MODE):
  - "srm": SRM-style high-pass residual (spatial domain)
  - "fft": log-magnitude FFT spectrum
Whichever holds up better across the transform grid wins as the
default; keep both implementations around either way -- the losing one
is still a useful row in the robustness/ablation table.

Also outputs the residual-energy scalar (see architecture doc) so the
fusion head has an explicit hint about how much this stream's output
can currently be trusted, rather than inferring it indirectly.

TODO (next step): implement both extraction modes, then a small
5-layer CNN (a few hundred thousand params) over the result, ending in
global average pooling -> 128-d embedding.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyStream(nn.Module):
    def __init__(self, mode: str = "srm"):
        super().__init__()

        if mode not in {"srm", "fft"}:
            raise ValueError(
                f"Unsupported frequency mode: {mode}. "
                "Use 'srm' or 'fft'."
            )

        self.mode = mode

        # Small CNN:
        # input: 3 x 224 x 224
        # output after global average pooling: 128-d
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def _extract_srm(self, image):
        """
        Simple SRM-style high-pass residual.

        image shape:
            [B, 3, H, W]

        Returns:
            residual [B, 3, H, W]
        """

        kernel = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=image.dtype,
            device=image.device,
        )

        kernel = kernel.view(1, 1, 3, 3)

        # Apply same high-pass filter independently
        # to R, G and B.
        kernel = kernel.repeat(3, 1, 1, 1)

        residual = F.conv2d(
            image,
            kernel,
            padding=1,
            groups=3,
        )

        return residual

    def _extract_fft(self, image):
        """
        Log-magnitude FFT representation.

        image shape:
            [B, 3, H, W]

        Returns:
            spectrum [B, 3, H, W]
        """

        fft = torch.fft.fft2(
            image,
            dim=(-2, -1),
        )

        fft = torch.fft.fftshift(
            fft,
            dim=(-2, -1),
        )

        magnitude = torch.abs(fft)

        spectrum = torch.log1p(magnitude)

        # Normalise each image/channel so the CNN
        # receives a more stable range.
        mean = spectrum.mean(
            dim=(-2, -1),
            keepdim=True,
        )

        std = spectrum.std(
            dim=(-2, -1),
            keepdim=True,
        )

        spectrum = (
            spectrum - mean
        ) / (std + 1e-6)

        return spectrum

    def encode(self, image):
        """
        Returns:
            embedding_128d:
                [B, 128]

            residual_energy_scalar:
                [B, 1]
        """

        if image.dim() == 3:
            image = image.unsqueeze(0)

        if self.mode == "srm":
            forensic_input = self._extract_srm(image)

        else:
            forensic_input = self._extract_fft(image)

        # Average absolute energy of the forensic signal
        residual_energy = (
            forensic_input
            .abs()
            .mean(dim=(1, 2, 3))
            .unsqueeze(1)
        )

        features = self.cnn(forensic_input)

        embedding = features.flatten(1)

        return embedding, residual_energy

    def forward(self, image):
        return self.encode(image)
