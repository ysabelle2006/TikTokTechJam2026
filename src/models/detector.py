"""
Top-level model: wires the spatial stream, frequency stream, and
fusion head into one callable that goes image -> confidence score.

This is the entry point a future deliverable/inference script should
use (per the architecture doc's §03 note: "for the deliverable script,
this whole pipeline is just a function from an image path to a float"
-- loop over a directory, call predict() per image, write out a JSON
list of {"image_path": ..., "pred": ...} records).

NOT used by train.py or evaluate.py directly, on purpose: both of those
read the SAME spatial embedding for an image many times (once per
epoch during training, once for the whole eval grid), and re-running
CLIP's forward pass live every time would reintroduce exactly the CPU
bottleneck cache_embeddings.py exists to avoid (see the architecture
doc's "Compute-aware order of operations" note). train.py's train_v1
reads pre-cached spatial embeddings directly instead; evaluate.py's
evaluate_v1 runs SpatialStream once per eval image because the eval
grid is small enough (13,500 images, once) that caching it wouldn't
pay for itself. Detector is for the "one image in, one score out,
no batching infrastructure needed" case those two scripts don't have.

Note on how a future train.py V3 (consistency loss) would use this
differently from V1's train_v1: infer.py-style scripts and evaluate.py
call predict() once per image. V3's training loop needs to call it (or
something with the same shared weights) TWICE per training example --
once on the clean image, once on a transformed copy -- so the
classification and consistency losses are computed against genuinely
shared-weight predictions rather than two separate models. That's not
built yet; V1 only needs a single forward pass per image.
"""

import torch

from models.frequency_stream import FrequencyStream
from models.fusion import FusionHead
from models.spatial_stream import SpatialStream


class Detector:
    def __init__(self, device: str = "cpu", freeze_spatial: bool = True, freq_mode: str = None):
        self.device = device
        self.spatial = SpatialStream(freeze=freeze_spatial, device=device)
        self.frequency = FrequencyStream(freeze=False, device=device, mode=freq_mode)
        self.fusion = FusionHead()
        self.fusion.to(device)

    def load_fusion_checkpoint(self, checkpoint_path):
        """Loads a checkpoint written by train.train_v1 (a dict with
        "frequency_cnn" and "fusion_head" state dicts). Puts both
        modules into eval mode -- callers doing inference only, not
        further training, should use this."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.frequency.model.load_state_dict(ckpt["frequency_cnn"])
        self.fusion.load_state_dict(ckpt["fusion_head"])
        self.frequency.model.eval()
        self.fusion.eval()

    @torch.no_grad()
    def predict(self, image) -> float:
        """PIL.Image (RGB) -> a raw fusion-head probability in [0, 1].

        NOTE: no calibration step is wired in yet (architecture doc
        §03's temperature-scaling plan) -- this is the uncalibrated
        sigmoid of the fusion logit, not a calibrated confidence.
        """
        spatial_tensor = self.spatial.prepare(image)
        spatial_embedding = self.spatial.encode(spatial_tensor)

        freq_tensor, energy = self.frequency.prepare(image)
        frequency_embedding = self.frequency.encode(freq_tensor)

        logit = self.fusion(spatial_embedding, frequency_embedding, energy)
        return torch.sigmoid(logit).item()
