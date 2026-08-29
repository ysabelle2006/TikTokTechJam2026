"""
Frequency stream: forensic residual + small CNN, trained from scratch.

Reads for the low-level statistical tell a generator leaves behind --
GAN upsampling checkerboards, diffusion denoising residuals -- which the
spatial stream is not looking for. This is the fragile signal: Gaussian
blur is a low-pass filter and removes exactly what this stream depends
on; heavy JPEG quantizes it away too.

Two interchangeable front-ends (config.FREQUENCY_MODE, see
transforms.preprocessing):
  - "srm": a bank of SRM-style high-pass residuals (spatial domain)
  - "fft": log-magnitude FFT spectrum
Both feed the same 5-layer CNN -> global average pool -> 128-d embedding.
Keep both around: the losing one is still a useful ablation row.

`encode` also returns the residual-energy scalar so the fusion head has
an explicit hint about how much to trust this stream right now.
"""

import torch
import torch.nn as nn

from tiktoktechjam2026 import config
from tiktoktechjam2026.transforms import preprocessing


def _conv_block(in_ch: int, out_ch: int, pool: bool, stride: int = 1) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class FrequencyStream(nn.Module):
    """
    ~0.4M params. Input channels depend on the mode (3 for srm, 1 for fft);
    output is a `config.FREQUENCY_EMBED_DIM`-d vector.
    """

    def __init__(self, mode: str = None):
        super().__init__()
        self.mode = mode or config.FREQUENCY_MODE
        in_ch = preprocessing.frequency_input_channels(self.mode)

        widths = (16, 32, 64, 128, config.FREQUENCY_EMBED_DIM)
        blocks = []
        prev = in_ch
        for i, w in enumerate(widths):
            # block 0 uses stride 2 (224 -> 112) to keep CPU cost down; blocks
            # 0..3 then pool: 112 -> 56 -> 28 -> 14 -> 7, block 4 keeps 7x7.
            blocks.append(_conv_block(prev, w, pool=(i < 4), stride=2 if i == 0 else 1))
            prev = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, 224, 224] frequency map -> [B, FREQUENCY_EMBED_DIM]."""
        if x.ndim == 3:
            x = x.unsqueeze(0)
        feats = self.features(x)
        return self.pool(feats).flatten(1)

    @torch.no_grad()
    def encode(self, image):
        """
        PIL image -> (embedding [FREQUENCY_EMBED_DIM], residual_energy float).

        Convenience path for single-image inference / explainability. Training
        and evaluate.py build batched frequency inputs via
        preprocessing.prepare_frequency_input and call forward() directly.
        """
        cnn_in = preprocessing.prepare_frequency_input(image, self.mode)
        emb = self.forward(cnn_in)[0]
        energy = preprocessing.residual_energy(image)
        return emb, energy
