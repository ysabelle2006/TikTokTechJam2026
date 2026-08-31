"""
Top-level model: wires the spatial stream, frequency stream, and
fusion head into one callable that goes image -> confidence score.

This is the entry point src/infer.py uses (per the architecture
doc's §03 note: "for the deliverable script, this whole pipeline is
just a function from an image path to a float" -- loop over a
directory, call predict() per image, write out a JSON list of
{"image_path": ..., "pred": ...} records).

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

import json
from pathlib import Path

import torch

from models.frequency_stream import FrequencyStream
from models.fusion import FusionHead, load_architecture_metadata
from models.spatial_stream import SpatialStream


class Detector:
    def __init__(self, device: str = "cpu", freeze_spatial: bool = True, freq_mode: str = None):
        self.device = device
        self.spatial = SpatialStream(freeze=freeze_spatial, device=device)
        self.frequency = FrequencyStream(freeze=False, device=device, mode=freq_mode)
        self.fusion = FusionHead()
        self.fusion.to(device)
        # 1.0 = no calibration applied (matches the old uncalibrated
        # behavior exactly). load_fusion_checkpoint() overwrites this if
        # it finds a calibration.json sitting next to the checkpoint.
        self.temperature = 1.0

    def load_fusion_checkpoint(self, checkpoint_path, calibration_path=None):
        """Loads a checkpoint written by train.train_v1/train_v2 (a dict
        with "frequency_cnn" and "fusion_head" state dicts). Puts both
        modules into eval mode -- callers doing inference only, not
        further training, should use this.

        Also looks for a calibration.json (written by src/calibrate.py)
        next to the checkpoint -- same directory, by default -- and
        loads its fitted temperature if present. Pass calibration_path
        explicitly to override that default location, or pass
        calibration_path=False to skip calibration entirely even if a
        file is sitting there (e.g. to deliberately compare calibrated
        vs. raw output). If no calibration file is found or loaded,
        self.temperature stays at 1.0 and predict() returns the same
        uncalibrated sigmoid it always did.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        # __init__ built a plain (non-gated) FusionHead before knowing
        # which checkpoint would be loaded -- rebuild it here if this
        # checkpoint's architecture.json (written by train.py alongside
        # a --use-freq-gate run) says otherwise, so load_state_dict below
        # sees a matching architecture instead of raising on missing/
        # unexpected "freq_gate.*" keys. Absent architecture.json (every
        # checkpoint saved before this option existed) -> False -> the
        # FusionHead() from __init__ is already correct, untouched.
        use_freq_gate = load_architecture_metadata(Path(checkpoint_path).parent)
        if use_freq_gate:
            self.fusion = FusionHead(use_freq_gate=True).to(self.device)
        self.frequency.model.load_state_dict(ckpt["frequency_cnn"])
        self.fusion.load_state_dict(ckpt["fusion_head"])
        self.frequency.model.eval()
        self.fusion.eval()

        if calibration_path is False:
            return
        if calibration_path is None:
            calibration_path = Path(checkpoint_path).parent / "calibration.json"
        calibration_path = Path(calibration_path)
        if calibration_path.is_file():
            with open(calibration_path) as f:
                self.temperature = json.load(f)["temperature"]
            print(f"Loaded calibration (temperature={self.temperature:.4f}) from {calibration_path}")
        else:
            print(f"No calibration file at {calibration_path} -- predict() will return an "
                  f"UNCALIBRATED sigmoid (run src/calibrate.py to fit one).")

    @torch.no_grad()
    def predict(self, image) -> float:
        """PIL.Image (RGB) -> a calibrated probability in [0, 1], if a
        calibration.json was loaded by load_fusion_checkpoint() -- see
        that method's docstring. Falls back to the plain, uncalibrated
        sigmoid of the fusion logit if none was found (self.temperature
        stays 1.0, which is a no-op: sigmoid(logit / 1.0) == sigmoid(logit))."""
        spatial_tensor = self.spatial.prepare(image)
        spatial_embedding = self.spatial.encode(spatial_tensor)

        freq_tensor, energy = self.frequency.prepare(image)
        frequency_embedding = self.frequency.encode(freq_tensor)

        logit = self.fusion(spatial_embedding, frequency_embedding, energy)
        return torch.sigmoid(logit / self.temperature).item()
