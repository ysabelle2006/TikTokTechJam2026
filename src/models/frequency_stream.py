"""
Frequency stream: high-pass/forensic residual + small CNN, trained from
scratch.

Reads for the low-level statistical tell a generator leaves behind --
GAN upsampling checkerboards, diffusion denoising residuals -- which
the spatial stream isn't looking for. This is the fragile signal:
Gaussian blur is a low-pass filter, so it removes exactly what this
stream depends on, and heavy JPEG quantizes it away too.

Deliberately NOT committed to one artifact-extraction method (see
transforms/preprocessing.py): "srm" (spatial-domain high-pass residual)
and "fft" (log-magnitude FFT spectrum) are both implemented, picked via
config.FREQUENCY_MODE or the `mode` argument here. Whichever holds up
better across the transform grid is a V1 finding, not an assumption --
run both (train.py / evaluate.py's --freq-mode flag) once the "srm"
default is confirmed working, and compare results/v1_fusion_srm.json
against results/v1_fusion_fft.json.

Also outputs the residual-energy scalar (see architecture doc) so the
fusion head has an explicit hint about how much this stream's output
can currently be trusted, rather than inferring it indirectly.

Unlike the spatial stream (models/spatial_stream.py), which reuses an
already-pretrained, frozen CLIP backbone, there's no pretrained model
for "is this pixel-level residual consistent with a generator" -- this
network has to learn that from scratch on our own train split. So:

  - freeze defaults to False (nothing pretrained to freeze; train.py's
    train_v1 backpropagates into this network every step)
  - encode() does NOT wrap the forward pass in torch.no_grad() -- the
    whole point is that gradients need to flow through here at
    training time. A caller doing pure inference (evaluate.py) wraps
    its own call site in `with torch.no_grad():` and/or sets
    freeze=True (which also calls .eval() on the BatchNorm layers, so
    they use their learned running stats instead of the eval batch's
    own statistics).

Interface note vs. SpatialStream: SpatialStream.prepare() returns just
a tensor, because CLIP's own preprocessing has no separate "reliability
scalar" to report. FrequencyStream.prepare() returns a (tensor, energy)
PAIR -- residual_energy is a plain numpy computation on the un-
normalized residual/spectrum map (see preprocessing.residual_energy),
not something the CNN produces, and not something that needs a
gradient, so it's computed once here rather than re-derived from the
embedding downstream.

Verified via scripts/preview_frequency_stream.py (which also confirms
transforms/preprocessing.py, see scripts/preview_frequency_input.py),
and exercised in production across every V1/V2 training/evaluation run
in results/ -- rerun that smoke test if you touch this module.
"""

import torch
import torch.nn as nn

from config import FREQUENCY_EMBED_DIM
from transforms.preprocessing import prepare_frequency_input, residual_energy


class _FrequencyCNN(nn.Module):
    """5 strided conv blocks + global average pool -> embed_dim.

    Widths 1->32->64->128->256->embed_dim land at ~0.68M params,
    matching the architecture doc's parameter-budget table
    ("Frequency-stream CNN (5 conv layers), ~0.6M params, fully
    trained"). Strides [2,2,2,2,1]: four downsampling steps take a
    224x224 input to 14x14, and the fifth conv stays at stride 1 (just
    widens the receptive field a bit further) rather than shrinking to
    an almost-featureless 7x7 before the CNN even gets to look at it --
    AdaptiveAvgPool2d(1) handles the final reduction to a single vector
    regardless of the exact spatial size that arrives at it.

    Starts with BatchNorm2d(1) rather than a hand-picked normalization
    constant. prepare_frequency_input() deliberately returns raw,
    un-renormalized residual/spectrum values (so residual_energy() can
    measure real cross-image amplitude differences -- see that
    function's docstring), which means whatever consumes its output
    still needs to land in a network-friendly range. A learned
    BatchNorm does that at the batch level instead of per-sample, so it
    doesn't wipe out the same amplitude signal a manual per-image
    min/max normalization would have.
    """

    def __init__(self, embed_dim: int = FREQUENCY_EMBED_DIM):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(1)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.features(x)
        return self.pool(x).flatten(1)


class FrequencyStream:
    def __init__(
        self,
        freeze: bool = False,
        device: str = "cpu",
        embed_dim: int = FREQUENCY_EMBED_DIM,
        mode: str = None,
    ):
        self.device = device
        self.frozen = freeze
        self.embed_dim = embed_dim
        self.mode = mode  # None -> preprocessing.py falls back to config.FREQUENCY_MODE

        model = _FrequencyCNN(embed_dim)
        model.to(device)
        if freeze:
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()

        self.model = model

    def prepare(self, pil_image):
        """PIL.Image (RGB) -> (tensor, residual_energy).

        tensor: shape (1, FREQUENCY_INPUT_SIZE, FREQUENCY_INPUT_SIZE),
        ready for encode(). residual_energy: a plain python float, the
        raw reliability scalar the fusion head consumes directly (see
        module docstring on why this returns a pair, unlike
        SpatialStream.prepare()).
        """
        freq_map = prepare_frequency_input(pil_image, mode=self.mode)
        energy = residual_energy(freq_map)
        tensor = torch.from_numpy(freq_map).unsqueeze(0)  # add channel dim -> (1, H, W)
        return tensor, energy

    def encode(self, image_tensor):
        """
        image_tensor: a single prepared tensor from prepare() (shape
        (1, H, W)), or a batch (N, 1, H, W). Returns a (embed_dim,)
        embedding for a single input, or (N, embed_dim) for a batch.

        Deliberately NOT wrapped in torch.no_grad() -- see module
        docstring.
        """
        single = image_tensor.dim() == 3
        batch = image_tensor.unsqueeze(0) if single else image_tensor
        batch = batch.to(self.device)

        features = self.model(batch)
        assert features.shape[-1] == self.embed_dim, (
            f"expected {self.embed_dim}-d embeddings, got {features.shape[-1]} -- "
            f"config.py's FREQUENCY_EMBED_DIM is out of sync with _FrequencyCNN"
        )
        return features.squeeze(0) if single else features
