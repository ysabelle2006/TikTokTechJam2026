"""
Top-level model: wires the spatial stream, frequency stream, and
fusion head into one callable that goes image -> confidence score.

This is the single entry point both training and inference should
use, so the two never drift apart.

Note on how this gets used differently by different scripts:
infer.py and evaluate.py call predict() once per image. train.py calls
it TWICE per training example -- once on the clean image, once on a
transformed copy -- reusing these exact same weights, so the
classification and consistency losses are computed against genuinely
shared-weight predictions rather than two separate models. See
train.py for the full objective.

TODO (after spatial_stream.py, frequency_stream.py, fusion.py exist):
implement Detector.predict(image) -> float end to end.
"""


import torch
from torchvision import transforms

from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.models.frequency_stream import FrequencyStream
from tiktoktechjam2026.models.fusion import FusionHead
from tiktoktechjam2026.transforms.preprocessing import prepare_spatial_input


class Detector:
    def __init__(self, frequency_mode="srm"):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.spatial_stream = SpatialStream(
            freeze=True
        )

        self.frequency_stream = FrequencyStream(
            mode=frequency_mode
        ).to(self.device)

        self.fusion_head = FusionHead().to(self.device)

        # Frequency branch should use an ordinary RGB tensor,
        # not CLIP-normalized values.
        self.frequency_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, image) -> float:
        image = image.convert("RGB")

        # Spatial branch
        spatial_input = prepare_spatial_input(image)
        spatial_input = spatial_input.unsqueeze(0)

        spatial_embedding = self.spatial_stream.encode(
            spatial_input
        )

        # Frequency branch
        frequency_input = self.frequency_preprocess(image)
        frequency_input = frequency_input.unsqueeze(0).to(
            self.device
        )

        frequency_embedding, residual_energy = (
            self.frequency_stream.encode(
                frequency_input
            )
        )

        # Fusion
        logit = self.fusion_head(
            spatial_embedding,
            frequency_embedding,
            residual_energy,
        )

        probability = torch.sigmoid(logit)

        return probability.item()