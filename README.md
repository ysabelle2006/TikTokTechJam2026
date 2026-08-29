# TikTokTechJam2026 -- Robust AIGC Image Detector

> A lightweight dual-stream AIGC detector combining spatial and
> forensic representations, trained with transformation-aware
> augmentation and explicit prediction consistency across transformed
> views.

One branch reads what the image depicts (a frozen CLIP backbone), the
other reads how it was made (a small CNN over a high-pass/frequency
residual). A fusion head combines both into a single confidence score.
Trained not just to survive JPEG compression, blur, resizing, noise,
color jitter, and cropping, but to keep agreeing with itself before and
after they happen.

Full architecture writeup: (link the published architecture doc here)

## Status

Scaffolding stage -- module structure and the experiment roadmap are in
place, implementations are being filled in step by step. See TODOs in
each file for what's next.

## Project structure

    src/tiktoktechjam2026/
      config.py              central config: image size, embedding dims, loss weight, paths
      transforms/
        augmentations.py     the 6 train-time robustness transforms from the brief
        preprocessing.py     per-stream input prep + the residual-energy scalar
      models/
        spatial_stream.py    frozen CLIP ViT-B/32 -> 512-d embedding
        frequency_stream.py  high-pass/SRM or FFT residual + small CNN -> 128-d embedding (ablation, see below)
        fusion.py            concat + MLP -> confidence score (V1; optional learned gate in V4)
        detector.py          wires the three pieces into one predict(image) call
      data/
        datasets.py          loaders for CIFAKE / SID_Set / WildFake
      cache_embeddings.py     offline CLIP feature-extraction + caching (CPU feasibility)
      train.py                training loop: classification + consistency loss (V3)
      evaluate.py             robustness/ablation evaluation harness
      infer.py                deliverable script: image dir -> JSON {image_path, pred}
      explain.py              Grad-CAM + frequency-spectrum explainability hooks
    results/                  one metrics file per roadmap stage, never overwritten

## Experiment roadmap: V0 -> V4

Build and evaluate in this order, keeping every stage's checkpoint and
metrics (see results/) -- this turns the project into a chain of
testable claims instead of one large thing that either works or
doesn't, and gives the write-up and pitch a much stronger narrative
than a single accuracy number.

  - **V0 -- spatial stream only**: baseline. Is CLIP + a small head
    already separating real from fake at all?
  - **V1 -- + frequency stream**: does the forensic branch add
    anything, or is the spatial stream carrying all the signal alone?
  - **V2 -- + transform-aware augmentation**: does training through the
    brief's transform grid shrink the clean-vs-transformed gap?
  - **V3 -- + consistency loss (our main proposed method)**: does
    explicitly penalizing disagreement between original and
    transformed predictions shrink that gap further than augmentation
    alone?
  - **V4 -- only if time allows**: learned gating between the two
    streams, and/or a short partial fine-tune of the backbone's last
    block.

Also run, as a separate ablation, frequency_stream.py's two modes
("srm" vs "fft") against the transform grid before committing to one
as the default.

| Detector               | Clean | JPEG | Blur | Resize | Avg robust drop |
|-------------------------|:-----:|:----:|:----:|:------:|:----------------:|
| Spatial only (V0)       |   --  |  --  |  --  |   --   |        --        |
| Frequency only          |   --  |  --  |  --  |   --   |        --        |
| Spatial + frequency (V1)|   --  |  --  |  --  |   --   |        --        |
| + augmentation (V2)     |   --  |  --  |  --  |   --   |        --        |
| + consistency (V3)      |   --  |  --  |  --  |   --   |        --        |

This table is the same one that satisfies the "Robustness Evaluation
Summary" deliverable -- fill it in as each stage finishes.

## Setup

    uv sync

(Requires network access to fetch torch and friends -- run this from a
regular terminal if it fails in a sandboxed shell.)

## Reproducing results

V0 and V1 are implemented. Both train a small head on a **frozen** CLIP
ViT-B/32 backbone, on the SID_Set `real` vs `full_synthetic` split
(deterministic 80/10/10, `config.SEED`). All transform parameters,
embedding dims, paths and the evaluation grid live in
`src/tiktoktechjam2026/config.py`.

```bash
# 0. fetch training data (real, multi-GB -- run from a normal terminal)
python downloads.py --sid --sid-cap 8000

# 1. build the deterministic train/val/test split
python -m tiktoktechjam2026.data.datasets --build

# 2. precompute frozen-CLIP embeddings (the CPU bottleneck) + V1 frequency
#    inputs. ~30 min on 8 CPU threads. train/val: clean only; test: all 15
#    robustness conditions.
python -m tiktoktechjam2026.cache_embeddings

# 3. V0 -- spatial stream only
python -m tiktoktechjam2026.train    --variant v0
python -m tiktoktechjam2026.evaluate --variant v0      # -> results/v0.json

# 4. V1 -- + frequency stream (SRM residual + CNN + residual-energy scalar)
python -m tiktoktechjam2026.train    --variant v1
python -m tiktoktechjam2026.evaluate --variant v1      # -> results/v1.json
```

`evaluate.py` prints the per-condition robustness table (Accuracy / Drop /
AUC) and writes it to `results/v{0,1}.json` -- one file per stage, never
overwritten.

Deliverable inference:

```bash
python -m tiktoktechjam2026.infer --images <dir> --variant v1 --out preds.json
```

Robustness transforms (`config.EVAL_CONDITIONS`) are applied to the raw
image at native resolution, before either stream's preprocessing; CLIP
normalization and the frequency residual are always recomputed from the
transformed pixels. Frequency-mode ablation (`srm` vs `fft`):
`train.py --variant v1 --freq-mode fft`.

## Limitations & what we'd improve with more time

TODO.

## Team contributions

TODO.
