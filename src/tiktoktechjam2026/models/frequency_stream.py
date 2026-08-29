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
    """
    SRM-style frequency / forensic stream.

    Input:
        image tensor of shape [B, 3, H, W], values roughly in [0, 1]

    Output:
        embedding:       [B, 128]
        residual_energy: [B, 1]
    """

    def __init__(self, mode: str = "srm"):
        super().__init__()

        if mode != "srm":
            raise ValueError("This implementation currently supports mode='srm' only.")

        self.mode = mode

        # ---------------------------------------------------------
        # 1. Fixed SRM-style high-pass filters
        # ---------------------------------------------------------

        # Horizontal / vertical local differences
        k1 = torch.tensor([
            [0.0,  0.0,  0.0],
            [0.0, -1.0,  1.0],
            [0.0,  0.0,  0.0],
        ])

        k2 = torch.tensor([
            [0.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  1.0,  0.0],
        ])

        # Laplacian-style high-pass filter
        k3 = torch.tensor([
            [0.0, -1.0,  0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0,  0.0],
        ])

        kernels = torch.stack([k1, k2, k3], dim=0).unsqueeze(1)

        # Fixed filters: not trainable
        self.register_buffer("srm_kernels", kernels)

        # ---------------------------------------------------------
        # 2. Small CNN over residual maps
        #
        # Input channels = 3 SRM residual maps
        # ---------------------------------------------------------

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def _to_grayscale(self, image):
        """
        Convert RGB tensor [B, 3, H, W] -> grayscale [B, 1, H, W].
        """

        if image.dim() == 3:
            image = image.unsqueeze(0)

        if image.shape[1] == 1:
            return image

        r = image[:, 0:1]
        g = image[:, 1:2]
        b = image[:, 2:3]

        gray = 0.299 * r + 0.587 * g + 0.114 * b

        return gray

    def extract_residuals(self, image):
        """
        Apply fixed SRM high-pass filters.

        Returns:
            residuals: [B, 3, H, W]
        """

        gray = self._to_grayscale(image)

        kernels = self.srm_kernels.to(
            device=gray.device,
            dtype=gray.dtype
        )

        residuals = F.conv2d(
            gray,
            kernels,
            padding=1
        )

        return residuals

    def encode(self, image):
        """
        Returns:
            embedding:       [B, 128]
            residual_energy: [B, 1]
        """

        residuals = self.extract_residuals(image)

        # Mean absolute high-frequency energy.
        # This gives fusion an explicit clue about how much
        # forensic/high-frequency signal remains in the image.
        residual_energy = residuals.abs().mean(
            dim=(1, 2, 3),
            keepdim=False
        ).unsqueeze(1)

        features = self.cnn(residuals)

        embedding = features.flatten(1)

        return embedding, residual_energy

    def forward(self, image):
        return self.encode(image)
