"""
Fusion head (V1): concatenates both streams' outputs and predicts
P(image is AI-generated).

Input:  [spatial_embedding (512-d), frequency_embedding (128-d),
         residual_energy_scalar (1-d)]  ->  641-d vector
Output: one raw logit (NOT sigmoided here -- see forward()'s docstring)

Honest framing, worth keeping in mind while training this: nothing here
*guarantees* the network learns to lean on the frequency stream when
it's intact and fall back on the spatial stream when it isn't -- that's
a plausible outcome of joint training, not a designed-in behavior. The
residual-energy scalar gives the MLP an explicit hint rather than
making it infer reliability purely from embedding statistics, but it's
still just an extra input feature, not a mechanism that forces that
behavior.

If V1-V3 (see architecture doc roadmap) show the network isn't actually
using that hint well, a natural upgrade (V4) is an explicit gate: a
small network conditioned on the residual-energy scalar that outputs a
weight in [0, 1] and multiplies the frequency embedding before fusion.

That diagnosis actually landed: evaluate.py's per-source FPR breakdown
on the v2_augmented_fft checkpoint showed real coco_val2017 images
misclassified as fake spike to 17-19% FPR under blur_sigma2.0/
resize_0.25 specifically (vs. a 2-4% baseline everywhere else, and vs.
sid_set's real images barely moving under the same transforms) -- the
signature of the frequency stream reading "flattened by heavy
blur/downsampling" as "flattened by generation" instead of falling back
on the spatial stream, exactly the failure mode this docstring warned
about. use_freq_gate below is that V4 upgrade, opt-in (default False)
so every existing V1/V2 checkpoint's architecture -- and therefore
FusionHead.load_state_dict() against it -- is completely unaffected
unless a caller explicitly asks for gating AND retrains with it.

Calibration (architecture doc §03: temperature scaling against a held-
out calibration split) is explicitly NOT wired in here -- this is the
raw fusion logit. AUC (the roadmap's primary metric) is threshold-free
and calibration-invariant, so it doesn't block V1's own question ("does
the forensic branch add anything"); calibration is a separate,
deliberately deferred post-processing step for whenever the project
gets to reporting an actual confidence/threshold, per the architecture
doc.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from config import FUSION_FREQ_GATE_HIDDEN_DIM, FUSION_HIDDEN_DIMS, FUSION_INPUT_DIM

ARCHITECTURE_FILENAME = "architecture.json"


def save_architecture_metadata(checkpoint_dir, use_freq_gate: bool) -> None:
    """Writes checkpoints/<dir>/architecture.json recording whether this
    checkpoint's FusionHead was built with use_freq_gate=True -- the
    state dict alone doesn't say so (a gated FusionHead's state dict
    just has extra "freq_gate.*" keys, which load_state_dict(strict=True)
    would refuse to match against a non-gated module's architecture, or
    silently ignore with strict=False -- neither is a safe way to
    *detect* which architecture to build before loading). Same sidecar
    pattern as calibrate.py's calibration.json: optional, additive,
    doesn't change what torch.save() puts in model.pt itself."""
    path = Path(checkpoint_dir) / ARCHITECTURE_FILENAME
    with open(path, "w") as f:
        json.dump({"use_freq_gate": use_freq_gate}, f, indent=2)


def load_architecture_metadata(checkpoint_dir) -> bool:
    """Returns the use_freq_gate flag saved by save_architecture_metadata
    for the checkpoint in checkpoint_dir, or False if no architecture.json
    is there -- which is exactly right for every checkpoint saved before
    this option existed (V0/V1/the original V2 run): they're all
    non-gated, so building a plain FusionHead() for them is correct, not
    just a fallback."""
    path = Path(checkpoint_dir) / ARCHITECTURE_FILENAME
    if not path.is_file():
        return False
    with open(path) as f:
        return bool(json.load(f).get("use_freq_gate", False))


class FusionHead(nn.Module):
    def __init__(self, input_dim: int = FUSION_INPUT_DIM, hidden_dims=FUSION_HIDDEN_DIMS,
                 use_freq_gate: bool = False, gate_hidden_dim: int = FUSION_FREQ_GATE_HIDDEN_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.use_freq_gate = use_freq_gate
        if use_freq_gate:
            # Small network conditioned ONLY on the residual-energy
            # scalar (per the module docstring's V4 sketch) -- outputs a
            # weight in [0, 1] that scales the frequency embedding
            # before it reaches the trunk below, instead of leaving the
            # trunk to infer on its own (per the docstring's "nothing
            # guarantees" warning) that a low-energy/degraded input
            # should count for less. Deliberately tiny: this is a gate,
            # not a second forensic branch -- it has one job.
            self.freq_gate = nn.Sequential(
                nn.Linear(1, gate_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid(),
            )
        dims = [input_dim, *hidden_dims]
        layers = []
        for in_d, out_d in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, spatial_embedding, frequency_embedding, residual_energy):
        """
        spatial_embedding: (N, 512) or (512,)
        frequency_embedding: (N, 128) or (128,)
        residual_energy: (N,) tensor, (N, 1) tensor, a 0-d tensor, or a
            plain python float -- whatever shape/type the caller has on
            hand; normalized to (N, 1) (or (1, 1) for a single example)
            before concatenation.

        Returns: raw logits, shape (N,) for a batch or a 0-d scalar
        tensor for a single example -- NOT sigmoided. Use
        nn.BCEWithLogitsLoss() for training and torch.sigmoid(...) at
        inference, matching train.V0Head's convention so V0 and V1
        checkpoints are used the same way.
        """
        single = spatial_embedding.dim() == 1
        if single:
            spatial_embedding = spatial_embedding.unsqueeze(0)
            frequency_embedding = frequency_embedding.unsqueeze(0)

        if not torch.is_tensor(residual_energy):
            residual_energy = torch.tensor(
                [residual_energy], dtype=spatial_embedding.dtype, device=spatial_embedding.device
            )
        residual_energy = residual_energy.to(dtype=spatial_embedding.dtype, device=spatial_embedding.device)
        if residual_energy.dim() == 0:
            residual_energy = residual_energy.view(1, 1)
        elif residual_energy.dim() == 1:
            residual_energy = residual_energy.unsqueeze(-1)  # (N,) -> (N, 1)

        if self.use_freq_gate:
            gate = self.freq_gate(residual_energy)  # (N, 1), in [0, 1]
            frequency_embedding = frequency_embedding * gate

        x = torch.cat([spatial_embedding, frequency_embedding, residual_energy], dim=-1)
        assert x.shape[-1] == self.input_dim, (
            f"expected a {self.input_dim}-d fused input, got {x.shape[-1]} -- check "
            f"SPATIAL_EMBED_DIM / FREQUENCY_EMBED_DIM / FUSION_INPUT_DIM in config.py"
        )

        logits = self.net(x).squeeze(-1)
        return logits.squeeze(0) if single else logits
