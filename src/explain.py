"""
Explainability hooks: two independent diagnostics, one per stream.

  --mode gradcam    Grad-CAM-style saliency overlay for the SPATIAL
                     stream (frozen CLIP ViT-B/32), for one image.
  --mode spectrum   Averaged FFT log-magnitude spectrum comparison for
                     the FREQUENCY stream, across four groups of
                     eval-grid images (real/fake x correct/incorrect).

Why two separate modes instead of one combined "explain this image"
view: the two streams read completely different signals (what the
image depicts vs. its low-level pixel statistics -- see
models/spatial_stream.py and models/frequency_stream.py's module
docstrings), so a single overlay can't honestly represent both at
once. Grad-CAM answers "which REGIONS of this one image moved the
score"; the spectrum comparison answers "what does the frequency
stream's INPUT look like, on average, for images the model gets right
vs. wrong" -- a distribution-level question that only makes sense
pooled over many images, not one.

--- Grad-CAM on a ViT, adapted ---

CLIP's ViT-B/32 has no classic conv "last layer" to hook the way
Grad-CAM normally does (a 2D conv feature map, channels x H x W) --
after the patch-embedding conv, everything is a sequence of patch
tokens processed by self-attention, not a spatial feature map. The
adaptation used here: hook the OUTPUT of the last transformer block
(visual.transformer.resblocks[-1]), which is a sequence of tokens (1
CLS token + one token per image patch). Grad-CAM's usual math --
channel weight = global-average-pooled gradient, CAM = ReLU(sum of
weight * activation) -- carries over directly if "spatial position" is
read as "patch token" instead of "pixel": the CLS token is dropped
(it's not tied to one image region), and the remaining 49 patch
tokens (for a 224x224 image with 32x32 patches, i.e. a 7x7 grid) are
reshaped back into that 7x7 grid, since open_clip's patch-embedding
conv output is (N, width, grid, grid) and gets flattened into that
same 49-token order before the transformer sees it. Bicubic-upsampled
to the original image size and colorized (see _overlay_cam) for a
normal-looking saliency heatmap.

Gradients need to flow through the (frozen) CLIP backbone for any of
this to work, which requires calling model.encode_image() directly
under grad-tracking (image_tensor.requires_grad_(True), no
torch.no_grad()) -- SpatialStream.encode() always wraps its forward
pass in @torch.no_grad() (see that module's docstring on why: it's
built for inference/caching only), so this deliberately bypasses it
and calls the underlying open_clip model directly instead. Freezing a
backbone's PARAMETERS (requires_grad=False on its weights) does not
block gradients from flowing back to an upstream tensor that itself
requires grad -- the chain rule doesn't care whether the weights along
the way need their own gradient, only whether the computation is
tracked at all, which the input tensor's requires_grad_(True) turns
on for this one forward pass.

CONFIRMED against a live open_clip install (3.3.0): runs end to end,
no shape-matching errors -- the axis-detection logic below (matching
each axis's SIZE against the known batch size, sequence length, and
feature dimension, rather than assuming a fixed (batch, seq, dim)
ordering) resolved correctly against this version's actual internal
layout.

What's NOT yet settled: the last block's heatmap can come out nearly
flat (little contrast even after normalizing to [0, 1]) for some
images -- confirmed on a real stylized/line-art example, where the
raw pre-normalization CAM had very low variance relative to its max.
Likely cause, not a bug: by the LAST transformer block, 11 rounds of
self-attention have already mixed every patch token with global
context from every other patch, which is a known limitation of
Grad-CAM on ViTs at the final layer -- spatial localization tends to
blur out exactly where the representation has become most
classification-relevant. --layer lets you hook an earlier block
instead (e.g. --layer 6 of 12 for ViT-B/32); _vit_gradcam also prints
the pre-normalization min/max/mean/std so low-contrast cases are
visible in the output rather than hidden behind normalization's
always-stretch-to-[0,1] behavior. There's no single correct layer to
default to here -- later is more relevant to the actual decision,
earlier is more spatially localized -- so this is left as something to
compare a few of, not something this script picks for you.

--- FFT spectrum comparison ---

Always uses FFT mode (transforms.preprocessing.prepare_frequency_input
with mode="fft") regardless of which mode the loaded checkpoint was
actually trained with -- this is a standalone forensic diagnostic
(are there periodic GAN/diffusion upsampling artifacts visible in the
spectrum?), not a rendering of what that specific checkpoint's
frequency stream saw. If you want to see what an "srm"-trained
checkpoint's stream actually received, that's prepare_frequency_input
mode="srm" instead, which this deliberately doesn't plot (a residual
map, not a spectrum, isn't "FFT spectrum plots" per the brief this
mode is answering).

Draws its image sample from data/eval_manifest.csv's clean-condition
rows (same images evaluate.py scores) -- reusing that grid means this
is asking the exact question the robustness table already contains
predictions for, not deriving its own separate sample.

Run with:  uv run python src/explain.py --mode gradcam --image path/to/image.jpg
           uv run python src/explain.py --mode spectrum --stage v2
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from config import CHECKPOINT_DIR, FREQUENCY_MODE
from models.detector import Detector
from transforms.preprocessing import prepare_frequency_input

# Same checkpoint-dir-prefix convention as evaluate.py/calibrate.py/infer.py.
CHECKPOINT_DIR_PREFIX = {"v1": "v1_fusion", "v2": "v2_augmented"}

EVAL_MANIFEST = Path("data/eval_manifest.csv")
OUTPUT_DIR = Path("outputs/explain")


def _resolve_checkpoint_path(stage: str, checkpoint_path: str = None, freq_mode: str = None) -> Path:
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
    else:
        mode_for_path = freq_mode or FREQUENCY_MODE
        checkpoint_path = Path(CHECKPOINT_DIR) / f"{CHECKPOINT_DIR_PREFIX[stage]}_{mode_for_path}" / "model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{checkpoint_path} doesn't exist -- run `uv run python src/train.py --stage {stage}` first "
            f"(add --freq-mode fft if you're pointing at the fft ablation)."
        )
    return checkpoint_path


def _load_detector(stage: str, checkpoint_path: str = None, freq_mode: str = None, calibrate: bool = True) -> Detector:
    checkpoint_path = _resolve_checkpoint_path(stage, checkpoint_path, freq_mode)
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
    resolved_mode = freq_mode or raw_ckpt.get("freq_mode") or FREQUENCY_MODE
    del raw_ckpt

    print("Loading CLIP backbone...")
    detector = Detector(device="cpu", freq_mode=resolved_mode)
    print(f"Loading frequency stream (mode={resolved_mode}) + fusion head from {checkpoint_path}...")
    detector.load_fusion_checkpoint(checkpoint_path, calibration_path=(None if calibrate else False))
    return detector


# ---------------------------------------------------------------- Grad-CAM


def _vit_gradcam(detector: Detector, image: Image.Image, layer_index: int = -1):
    """Returns (cam, pred): cam is a (grid, grid) float32 numpy array in
    [0, 1] (grid=7 for ViT-B/32 at 224x224), pred is the model's
    (calibrated, if loaded) probability for `image`. See module
    docstring for the method and its unverified-layout caveat.

    layer_index selects which of visual.transformer.resblocks to hook
    (Python indexing, so -1 is the last block, 0 the first). Confirmed
    against a live install: the code runs end to end, but the LAST
    block's heatmap can come out nearly flat (little contrast after
    normalizing) for some images -- by the final block, many rounds of
    self-attention have mixed every patch token with global context
    from every other patch, which is known to blur out Grad-CAM's
    spatial localization on ViTs. A middle block (e.g. layer_index=6 of
    12 for ViT-B/32) usually still carries patch-local signal and often
    localizes better -- try a few if the default looks uninformative,
    there's no single "correct" layer for this, it's a genuine
    trade-off between how class-relevant the features are (later =
    more relevant to the actual decision) and how spatially localized
    they still are (earlier = more localized)."""
    model = detector.spatial.model
    visual = model.visual
    captured = {}

    def fwd_hook(module, inp, out):
        activation = out[0] if isinstance(out, (tuple, list)) else out
        activation.retain_grad()
        captured["activation"] = activation

    handle = visual.transformer.resblocks[layer_index].register_forward_hook(fwd_hook)
    try:
        image_tensor = detector.spatial.prepare(image).unsqueeze(0)
        image_tensor.requires_grad_(True)

        # Bypass SpatialStream.encode()'s @torch.no_grad() wrapper --
        # need gradients to flow back to the hooked activation. See
        # module docstring for why frozen weights don't block this.
        spatial_embedding = model.encode_image(image_tensor).squeeze(0)

        with torch.no_grad():
            freq_tensor, energy = detector.frequency.prepare(image)
            frequency_embedding = detector.frequency.encode(freq_tensor)

        logit = detector.fusion(spatial_embedding, frequency_embedding, energy)
        pred = torch.sigmoid(logit / detector.temperature)

        logit.backward()

        if "activation" not in captured or captured["activation"].grad is None:
            raise RuntimeError(
                "Grad-CAM hook didn't capture a gradient on the last transformer block's "
                "output. open_clip's internal structure may not match what this function "
                "expects -- see this module's docstring."
            )
        activation = captured["activation"]
        grad = activation.grad

        patch_stride = visual.conv1.stride[0]
        input_size = image_tensor.shape[-1]
        grid = input_size // patch_stride
        seq_len = grid * grid + 1  # +1 for the CLS/class token
        feature_dim = visual.conv1.out_channels

        shape = list(activation.shape)
        if len(shape) != 3:
            raise RuntimeError(
                f"expected the last transformer block's output to be a 3-D tensor "
                f"(batch, seq, dim) in some order, got shape {tuple(activation.shape)}."
            )

        def _find_axis(size, used):
            for i, s in enumerate(shape):
                if s == size and i not in used:
                    return i
            return None

        used = set()
        batch_axis = _find_axis(1, used)
        if batch_axis is not None:
            used.add(batch_axis)
        seq_axis = _find_axis(seq_len, used)
        if seq_axis is not None:
            used.add(seq_axis)
        feat_axis = _find_axis(feature_dim, used)
        if feat_axis is not None:
            used.add(feat_axis)
        if None in (batch_axis, seq_axis, feat_axis) or len(used) != 3:
            raise RuntimeError(
                f"couldn't map the captured activation's shape {tuple(activation.shape)} onto "
                f"(batch=1, seq_len={seq_len}, feature_dim={feature_dim}) -- open_clip's internal "
                f"tensor layout for this installed version may differ from what this function "
                f"assumes. See this module's docstring's Grad-CAM caveat."
            )

        # Reorder to canonical (batch, seq, dim), regardless of the
        # library's actual internal ordering.
        activation_bsf = activation.permute(batch_axis, seq_axis, feat_axis)[0]  # (seq_len, feature_dim)
        grad_bsf = grad.permute(batch_axis, seq_axis, feat_axis)[0]

        patch_activations = activation_bsf[1:]  # drop the CLS token -> (n_patches, feature_dim)
        patch_grads = grad_bsf[1:]

        channel_weights = patch_grads.mean(dim=0)  # (feature_dim,) -- GAP over patches, per Grad-CAM
        cam = torch.relu((patch_activations * channel_weights).sum(dim=-1))  # (n_patches,)
        cam = cam.reshape(grid, grid).detach()  # backward() already ran -- nothing past this point needs grad

        # Report the PRE-normalization spread -- normalizing by max
        # always stretches the output to [0, 1] even if the real signal
        # is nearly flat, which would otherwise silently hide a
        # near-uninformative heatmap behind what looks like full
        # contrast. raw_std small relative to raw_mean is the signature
        # of that -- worth checking before trusting the picture.
        raw_min, raw_max = float(cam.min()), float(cam.max())
        raw_mean, raw_std = float(cam.mean()), float(cam.std())
        print(f"pre-normalization CAM stats (layer {layer_index}): "
              f"min={raw_min:.6g} max={raw_max:.6g} mean={raw_mean:.6g} std={raw_std:.6g}")
        if raw_max > 0 and raw_std / raw_max < 0.05:
            print("NOTE: std is small relative to max -- this heatmap has little spatial contrast "
                  "before normalization stretches it to look like full contrast. Try a different "
                  "--layer (e.g. a middle block) before reading much into this one.")

        cam = cam / (raw_max + 1e-8)
        return cam.detach().cpu().numpy().astype(np.float32), float(pred.item())
    finally:
        handle.remove()


def _overlay_cam(image: Image.Image, cam: np.ndarray) -> Image.Image:
    """cam: (grid, grid) float array in [0, 1]. Bicubic-upsamples to
    image's size, applies a jet colormap, and alpha-blends over the
    original image -- a standard Grad-CAM-style overlay."""
    w, h = image.size
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    blended = np.clip(0.55 * base + 0.45 * heatmap_rgb.astype(np.float32), 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def run_gradcam(image_path: str, stage: str = "v2", checkpoint_path: str = None,
                 freq_mode: str = None, calibrate: bool = True, output_path: str = None,
                 layer_index: int = -1):
    detector = _load_detector(stage, checkpoint_path, freq_mode, calibrate)
    image = Image.open(image_path).convert("RGB")

    cam, pred = _vit_gradcam(detector, image, layer_index=layer_index)

    output_path = Path(output_path) if output_path else OUTPUT_DIR / f"{Path(image_path).stem}_gradcam_layer{layer_index}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = _overlay_cam(image, cam)
    overlay.save(output_path)

    print(f"\npred (P[fake]): {pred:.4f}")
    print(f"Wrote {output_path}")
    return {"image_path": image_path, "pred": pred, "output_path": str(output_path)}


# ---------------------------------------------------- FFT spectrum comparison


def _load_eval_manifest():
    if not EVAL_MANIFEST.is_file():
        raise FileNotFoundError(f"{EVAL_MANIFEST} doesn't exist -- run `python scripts/build_eval_grid.py` first.")
    with open(EVAL_MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def _colorize(map2d: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 255] and apply a viridis colormap --
    just for a readable heatmap, no claim that absolute magnitude is
    comparable across groups beyond what min-max already shows."""
    m = map2d - map2d.min()
    peak = m.max()
    if peak > 0:
        m = m / peak
    colored_bgr = cv2.applyColorMap(np.uint8(255 * m), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def _compose_spectrum_grid(group_maps: dict, cell_size: int, output_path: Path):
    """group_maps: {"real_correct": (avg_map, n), ...}, avg_map is None
    if that group had zero images. Renders a labeled 2x2 grid."""
    pad, label_h = 24, 22
    canvas_w = cell_size * 2 + pad * 3
    canvas_h = cell_size * 2 + pad * 3 + label_h * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    layout = [("real_correct", 0, 0), ("fake_correct", 1, 0),
              ("real_incorrect", 0, 1), ("fake_incorrect", 1, 1)]
    for key, col, row in layout:
        x = pad + col * (cell_size + pad)
        y = pad + label_h + row * (cell_size + pad + label_h)
        avg_map, n = group_maps[key]
        draw.text((x, y - label_h), f"{key}  (n={n})", fill="black")
        if avg_map is None:
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline="black")
            draw.text((x + cell_size // 2 - 30, y + cell_size // 2), "no images", fill="black")
        else:
            tile = Image.fromarray(_colorize(avg_map)).resize((cell_size, cell_size), Image.BICUBIC)
            canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_spectrum_comparison(stage: str = "v2", checkpoint_path: str = None, freq_mode: str = None,
                             calibrate: bool = True, n_per_group: int = 200,
                             cell_size: int = 256, output_path: str = None):
    detector = _load_detector(stage, checkpoint_path, freq_mode, calibrate)

    rows = _load_eval_manifest()
    clean_rows = [r for r in rows if r["condition"] == "clean"]
    print(f"{len(clean_rows)} clean-condition rows in {EVAL_MANIFEST}, scoring up to "
          f"{n_per_group} images per group (real/fake x correct/incorrect)...")

    group_sums = {k: None for k in ("real_correct", "real_incorrect", "fake_correct", "fake_incorrect")}
    group_counts = {k: 0 for k in group_sums}

    for r in tqdm(clean_rows, desc="scoring + accumulating spectra"):
        if all(group_counts[k] >= n_per_group for k in group_counts):
            break
        actually_fake = int(r["label"]) == 1
        base_key = "fake" if actually_fake else "real"
        # Skip this row entirely if BOTH of its possible groups
        # (correct/incorrect) are already full -- cheaper than scoring
        # an image we can't use either outcome of.
        if group_counts[f"{base_key}_correct"] >= n_per_group and group_counts[f"{base_key}_incorrect"] >= n_per_group:
            continue

        image = Image.open(r["transformed_path"]).convert("RGB")
        pred = detector.predict(image)
        predicted_fake = pred > 0.5
        correct = predicted_fake == actually_fake
        key = f"{base_key}_{'correct' if correct else 'incorrect'}"
        if group_counts[key] >= n_per_group:
            continue

        # Always fft here, independent of the checkpoint's own
        # freq_mode -- see module docstring.
        spectrum = prepare_frequency_input(image, mode="fft")
        group_sums[key] = spectrum if group_sums[key] is None else group_sums[key] + spectrum
        group_counts[key] += 1

    group_maps = {}
    for key, total in group_sums.items():
        n = group_counts[key]
        group_maps[key] = (None if n == 0 else total / n, n)
        print(f"{key}: n={n}")

    output_path = Path(output_path) if output_path else OUTPUT_DIR / "spectrum_comparison.png"
    _compose_spectrum_grid(group_maps, cell_size, output_path)
    print(f"\nWrote {output_path}")
    return {k: n for k, (_, n) in group_maps.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["gradcam", "spectrum"])
    parser.add_argument("--image", default=None, help="gradcam mode: path to a single image")
    parser.add_argument("--stage", default="v2", choices=["v1", "v2"])
    parser.add_argument("--checkpoint", default=None, help="override the default checkpoint path")
    parser.add_argument("--freq-mode", default=None, choices=["srm", "fft"],
                         help="defaults to the checkpoint's own recorded mode")
    parser.add_argument("--no-calibration", action="store_true",
                         help="skip loading calibration.json even if present")
    parser.add_argument("--n-per-group", type=int, default=200,
                         help="spectrum mode: max images averaged per real/fake x correct/incorrect group")
    parser.add_argument("--layer", type=int, default=-1,
                         help="gradcam mode: which visual.transformer.resblocks index to hook "
                              "(-1 = last block, the default; try a middle block, e.g. 6 of 12 "
                              "for ViT-B/32, if the default heatmap looks flat -- see this "
                              "module's docstring)")
    parser.add_argument("--output", default=None, help="where to write the output PNG")
    args = parser.parse_args()

    if args.mode == "gradcam":
        if not args.image:
            parser.error("--mode gradcam requires --image")
        run_gradcam(
            image_path=args.image,
            stage=args.stage,
            checkpoint_path=args.checkpoint,
            freq_mode=args.freq_mode,
            calibrate=not args.no_calibration,
            output_path=args.output,
            layer_index=args.layer,
        )
    else:
        run_spectrum_comparison(
            stage=args.stage,
            checkpoint_path=args.checkpoint,
            freq_mode=args.freq_mode,
            calibrate=not args.no_calibration,
            n_per_group=args.n_per_group,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
