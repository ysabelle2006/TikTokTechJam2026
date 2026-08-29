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
Don't build this before V1-V3 exist -- there's nothing to diagnose it
against yet.

Calibration (architecture doc §03: temperature scaling against a held-
out calibration split) is explicitly NOT wired in here -- this is the
raw fusion logit. AUC (the roadmap's primary metric) is threshold-free
and calibration-invariant, so it doesn't block V1's own question ("does
the forensic branch add anything"); calibration is a separate,
deliberately deferred post-processing step for whenever the project
gets to reporting an actual confidence/threshold, per the architecture
doc.
"""

import torch
import torch.nn as nn

from config import FUSION_HIDDEN_DIMS, FUSION_INPUT_DIM


class FusionHead(nn.Module):
    def __init__(self, input_dim: int = FUSION_INPUT_DIM, hidden_dims=FUSION_HIDDEN_DIMS):
        super().__init__()
        self.input_dim = input_dim
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

        x = torch.cat([spatial_embedding, frequency_embedding, residual_energy], dim=-1)
        assert x.shape[-1] == self.input_dim, (
            f"expected a {self.input_dim}-d fused input, got {x.shape[-1]} -- check "
            f"SPATIAL_EMBED_DIM / FREQUENCY_EMBED_DIM / FUSION_INPUT_DIM in config.py"
        )

        logits = self.net(x).squeeze(-1)
        return logits.squeeze(0) if single else logits
