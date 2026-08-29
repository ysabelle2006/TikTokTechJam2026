"""
Top-level model: wires the spatial stream, frequency stream, and fusion
head into one object that goes image -> P(AI-generated).

Single source of truth for both training and inference, so the two never
drift:

  * infer.py / evaluate.py call `predict` (or the batched `classify`) once
    per image.
  * train.py calls `classify` on cached CLIP embeddings (V0) or on cached
    embeddings + live frequency maps (V1). The classification path is the
    same code either way.

Variants:
  v0 : frozen CLIP  ->  SpatialHead            -> logit
  v1 : frozen CLIP  ->  \
       SRM/FFT map  ->  FrequencyStream CNN ->  FusionHead -> logit
                        residual-energy scalar /
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tiktoktechjam2026 import config
from tiktoktechjam2026.models.frequency_stream import FrequencyStream
from tiktoktechjam2026.models.fusion import FusionHead
from tiktoktechjam2026.models.spatial_head import SpatialHead
from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.transforms import preprocessing

VARIANTS = ("v0", "v1")


class Detector(nn.Module):
    def __init__(self, variant: str = "v0", freq_mode: str = None, device: str = None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.freq_mode = freq_mode or config.FREQUENCY_MODE
        self.device = torch.device(device or config.DEVICE)

        self.spatial_stream = SpatialStream()
        if variant == "v0":
            self.spatial_head = SpatialHead()
        else:
            self.frequency_stream = FrequencyStream(self.freq_mode)
            self.fusion_head = FusionHead()

        self.to(self.device)

    # ------------------------------------------------------------------
    # Trainable parameters (CLIP is always frozen for V0/V1)
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        if self.variant == "v0":
            return self.spatial_head.parameters()
        return list(self.frequency_stream.parameters()) + list(self.fusion_head.parameters())

    def _head_modules(self) -> dict[str, nn.Module]:
        """The parts we actually checkpoint (everything except frozen CLIP)."""
        if self.variant == "v0":
            return {"spatial_head": self.spatial_head}
        return {"frequency_stream": self.frequency_stream, "fusion_head": self.fusion_head}

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _as_pil_list(self, images):
        from PIL import Image
        if isinstance(images, Image.Image):
            return [images]
        return list(images)

    @torch.no_grad()
    def embed_spatial(self, images) -> torch.Tensor:
        """PIL image / list of PIL images -> [B, 512] CLIP embedding (on device)."""
        pil = self._as_pil_list(images)
        batch = torch.stack([preprocessing.prepare_spatial_input(im) for im in pil])
        return self.spatial_stream.encode(batch.to(self.device))

    def frequency_batch(self, images):
        """
        PIL image / list -> (freq_input [B, C, 224, 224], residual_energy [B]),
        both on device. Used by train.py (V1) and evaluate.py.
        """
        pil = self._as_pil_list(images)
        freq = torch.stack([
            preprocessing.prepare_frequency_input(im, self.freq_mode) for im in pil
        ]).to(self.device)
        energy = torch.tensor(
            [preprocessing.residual_energy(im) for im in pil],
            dtype=torch.float32, device=self.device,
        )
        return freq, energy

    # ------------------------------------------------------------------
    # Classification (shared by training and inference)
    # ------------------------------------------------------------------
    def classify(
        self,
        spatial_embedding: torch.Tensor,
        freq_input: torch.Tensor = None,
        residual_energy: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        spatial_embedding: [B, 512] (from embed_spatial or the offline cache)
        freq_input / residual_energy: required for v1, ignored for v0
        Returns: [B] raw logits.
        """
        spatial_embedding = spatial_embedding.to(self.device)
        if self.variant == "v0":
            return self.spatial_head(spatial_embedding)

        if freq_input is None or residual_energy is None:
            raise ValueError("v1 classify() needs freq_input and residual_energy")
        freq_emb = self.frequency_stream(freq_input.to(self.device))
        return self.fusion_head(spatial_embedding, freq_emb, residual_energy.to(self.device))

    # ------------------------------------------------------------------
    # Inference entry points
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, images) -> np.ndarray:
        """PIL image / list -> np.ndarray of P(AI-generated) in [0, 1]."""
        self.eval()
        spatial = self.embed_spatial(images)
        if self.variant == "v0":
            logits = self.classify(spatial)
        else:
            freq, energy = self.frequency_batch(images)
            logits = self.classify(spatial, freq, energy)
        return torch.sigmoid(logits).cpu().numpy()

    def predict(self, image) -> float:
        """Single PIL image -> scalar P(AI-generated)."""
        return float(self.predict_proba([image])[0])

    # ------------------------------------------------------------------
    # Checkpointing (heads only -- CLIP weights come from open_clip)
    # ------------------------------------------------------------------
    def save(self, path: str, meta: dict = None):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        blob = {
            "variant": self.variant,
            "freq_mode": self.freq_mode,
            "state": {k: m.state_dict() for k, m in self._head_modules().items()},
            "meta": meta or {},
        }
        torch.save(blob, path)

    def load(self, path: str):
        blob = torch.load(path, map_location=self.device)
        if blob["variant"] != self.variant:
            raise ValueError(
                f"checkpoint is {blob['variant']!r}, detector is {self.variant!r}"
            )
        for k, m in self._head_modules().items():
            m.load_state_dict(blob["state"][k])
        self.to(self.device)
        return blob.get("meta", {})

    @classmethod
    def from_checkpoint(cls, path: str, device: str = None) -> "Detector":
        blob = torch.load(path, map_location="cpu")
        det = cls(variant=blob["variant"], freq_mode=blob.get("freq_mode"), device=device)
        det.load(path)
        return det
