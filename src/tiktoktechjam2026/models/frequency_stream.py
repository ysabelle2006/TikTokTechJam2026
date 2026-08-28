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


class FrequencyStream:
    def __init__(self, mode: str = "srm"):
        raise NotImplementedError

    def encode(self, image):
        """Returns (embedding_128d, residual_energy_scalar)."""
        raise NotImplementedError
