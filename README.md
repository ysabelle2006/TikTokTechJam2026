# TikTokTechJam2026 -- Robust AIGC Image Detector

> A lightweight dual-stream AIGC detector combining spatial and
> forensic representations, trained with transformation-aware
> augmentation so accuracy holds up after the redistribution
> pipeline (compression, cropping, resizing, blur, noise, color
> shifts) that real-world images actually go through.

One branch reads what the image depicts (a frozen CLIP ViT-B/32
backbone), the other reads how it was made (a small CNN over a
frequency-domain residual). A fusion head combines both into a single
confidence score, calibrated so that score is an actual probability
rather than a raw, typically overconfident sigmoid.

## Status

V0 (spatial-only baseline), V1 (+ frequency stream, with an "srm" vs
"fft" ablation), and V2 (+ transform-aware augmentation, with a fixed
checkpoint-selection bug from V1) are all implemented, trained, and
evaluated -- see the roadmap table below for real numbers. V3
(consistency loss between a clean and a transformed prediction of the
SAME model) is designed but **not implemented**: V2's full
cache+train+evaluate cycle already took roughly 12 hours end-to-end on
the development machine, and V3's training step is strictly more
expensive per step (two forward passes per example instead of one).
Given the hackathon time budget, we chose to stop at V2 and spend the
remaining time on the deliverable pieces every submission needs
(inference script, calibration, explainability, error analysis)
rather than gamble that time on a second multi-hour training run. See
"Limitations & what we'd improve" below for the honest trade-off this
represents. V4's originally-scoped form (learned gating AND a partial
backbone fine-tune) was not attempted in full, but a narrower,
data-driven slice of it was: after `evaluate.py --dump-predictions`
+ `error_analysis.py` identified two specific conditions
(`blur_sigma2.0`, `resize_0.25`) where real `coco_val2017` images were
being misclassified as fake, we added an opt-in residual-energy gate
on the frequency embedding (`FusionHead(use_freq_gate=True)`) plus
upweighted sampling of those two conditions during training, retrained
V2 with both changes, and evaluated the result as a separate,
non-destructive checkpoint (`v2_augmented_fft_gated`) rather than
overwriting the baseline. See "Ablation: frequency gate +
failure-condition reweighting" below for the full result -- short
version: it fixed the targeted problem via a genuine AUC improvement,
but cost accuracy elsewhere (broader FNR increase, a small dip in the
held-out-generator AUC), so we're keeping it as a documented
alternative rather than replacing V2 with it for submission -- see
"Submission model" directly below for the reasoning, argued in the
brief's own terms.

## Submission model

**We're submitting the baseline V2 checkpoint** --
`checkpoints/v2_augmented_fft/model.pt`, logged at
`results/v2_augmented_fft.json` -- **not** the frequency-gate +
reweighting ablation described further down. `infer.py` uses it by
default (`--stage v2`, no extra flags needed).

Short version, in the brief's own terms (full numbers in "Ablation:
frequency gate + failure-condition reweighting" below): the gated
checkpoint scores a little higher on our own internal robustness
tracking, but that tracking is computed entirely on `validation_demo`
-- the reference/demo benchmark section 5.4 explicitly says "will not
contribute to the final score." What the gate actually does is trade a
real fix for **robustness** (fewer real images misflagged under
blur/resize) for a small but genuine cost to **generalization** (the
held-out-generator AUC drops 0.9559 -> 0.9527). Section 1 of the
brief's own "Core Challenge" slide names generalization to new
generators as the harder, more central problem -- ahead of robustness
to transforms -- so we kept the checkpoint that doesn't trade away
performance on the axis the brief treats as hardest, and reported the
gate as a fully-measured, honestly-discussed experiment instead of
quietly picking whichever number was larger. That's also exactly the
"thoughtful discussion of trade-offs such as robustness,
generalisation, and false positives" section 5.2 asks for.

## Project structure

    src/
      config.py              central config: image size, embedding dims, loss weight, paths
      transforms/
        augmentations.py     the brief's 6 transform families + a named-condition registry
        preprocessing.py     per-stream input prep + the residual-energy scalar
      models/
        spatial_stream.py    frozen CLIP ViT-B/32 -> 512-d embedding
        frequency_stream.py  high-pass/SRM or FFT residual + small CNN -> 128-d embedding (ablation, see below)
        fusion.py            concat + MLP -> confidence score
        detector.py          wires the three pieces into one predict(image) call, calibration-aware
      data/
        datasets.py          loaders for CIFAKE / SID_Set / WildFake / COCO val2017
      cache_embeddings.py     offline CLIP feature-extraction + caching (CPU feasibility)
      train.py                training loop: V0 (spatial-only), V1 (+ frequency), V2 (+ augmentation)
      evaluate.py             robustness evaluation harness: AUC + Accuracy/FPR/FNR per condition
      calibrate.py            fits a temperature-scaling calibration on a held-out split
      infer.py                THE deliverable script: image dir -> JSON {image_path, pred}
      explain.py              Grad-CAM (spatial stream) + FFT spectrum comparison (frequency stream)
      error_analysis.py       pulls misclassified examples, diagnoses spatial-only vs. fusion-induced errors
    scripts/                  one-off utilities: build_eval_grid.py, preview_*.py smoke tests
    checkpoints/              one directory per trained stage (model.pt + calibration.json)
    results/                  one metrics file per roadmap stage, never overwritten
    outputs/                  infer.py / explain.py output (predictions.json, Grad-CAM overlays, etc.)

`data/`, `cache/`, `checkpoints/`, and `outputs/` are all gitignored --
none of them are in this repo. `results/` (metrics + error-analysis
files, not the large per-row prediction CSVs) IS tracked, so you can
read our actual numbers without rerunning anything. See "Data" below
for how to put `data/` back together before reproducing a training run
yourself.

## Experiment roadmap: V0 -> V4

Built and evaluated in this order, keeping every stage's checkpoint and
metrics (see `results/`) -- this turns the project into a chain of
testable claims instead of one large thing that either works or
doesn't.

  - **V0 -- spatial stream only** (done): baseline. Is CLIP + a small
    head already separating real from fake at all? Yes, clearly above
    chance, but with the largest clean-vs-robust gap of any stage.
  - **V1 -- + frequency stream** (done): does the forensic branch add
    anything? Yes -- adding it lifts both clean and robust AUC over
    V0. Also ran the "srm" vs "fft" ablation here (see below);
    **"fft" is the current default** (`config.FREQUENCY_MODE`).
  - **V2 -- + transform-aware augmentation** (done): does training
    through the brief's transform grid shrink the clean-vs-transformed
    gap? Yes, meaningfully -- the avg robust drop roughly halves
    versus V1(fft). Also fixed a real checkpoint-selection blind spot
    from V1 along the way (see Limitations).
  - **V3 -- + consistency loss (our main proposed method)** (**not
    implemented** -- see Status/Limitations): would explicitly
    penalize disagreement between a clean and a transformed prediction
    of the same model, on top of augmentation alone. This was the
    project's actual novelty claim beyond a fairly standard
    spatial+frequency fusion architecture; scoping it out was a
    deliberate time-budget decision, not an oversight, and is
    discussed honestly below.
  - **V4 -- only if time allows** (not attempted): learned gating
    between the two streams, and/or a short partial fine-tune of the
    backbone's last block.

### Robustness Evaluation Summary

| Detector                          | Clean AUC | Avg Robust AUC | Avg Robust Drop | Final Score |
|------------------------------------|:---------:|:--------------:|:----------------:|:-----------:|
| V0 -- spatial only                 |  0.9529   |     0.9244      |      0.0285       |   0.9387    |
| V1 -- + frequency ("srm")          |  0.9684   |     0.9462      |      0.0223       |   0.9573    |
| V1 -- + frequency ("fft", default) |  0.9738   |     0.9528      |      0.0210       |   0.9633    |
| **V2 -- + augmentation ("fft")**   |**0.9702** |   **0.9586**    |    **0.0116**     | **0.9644**  |
| V2 + freq. gate + reweighting (ablation) | 0.9706 | 0.9612 | 0.0095 | 0.9659 |

The gated row is not a strict win despite the higher Final Score --
see the ablation section below for why we did not promote it to the
"current best" row above it.

Final Score = 0.50 x Clean AUC + 0.50 x Avg Robust AUC, matching the
brief. `evaluate.py` also reports Accuracy/FPR/FNR at a fixed 0.5
decision threshold alongside AUC, per condition (see its module
docstring for why 0.5 is a fair choice with or without calibration
applied) -- that field was added after V0/V1's `results/*.json` were
last generated, so only `results/v2_augmented_fft.json` currently has
it; re-running `evaluate.py --stage v0` / `--stage v1` would backfill
it for those (a quick re-evaluation, not a re-training).

**What this table does and doesn't determine.** Every row above (and
the held-out-generator check further down) is computed against
`validation_demo` -- which matches the reference benchmark the brief
describes in section 5.4 almost exactly: our `coco_val2017` (5,000
images) and `wildfake_dalle`/`dalle_advanced` (8,844 images) line up
with its stated 4,998 and 8,843. Section 5.4 is explicit that this set
"serves only as a reference benchmark and will not contribute to the
final score." So this table satisfies the required Robustness
Evaluation Summary deliverable and is how we tracked progress across
V0-V2 -- it is not a literal scored leaderboard number. We still treat
it as the most important evidence we have, because it's the only
labelled signal available for the two things the brief's own "Core
Challenge" slide names as what actually makes this hard (generalizing
to new generators, staying robust to redistribution transforms) -- see
"Submission model" above for how that distinction actually shaped
which checkpoint we chose to submit.

**Comparability caveat, stated plainly rather than glossed over**: V0
and V1 were evaluated against a 15-condition grid; V2 was evaluated
against a 19-condition grid (4 compound/stacked conditions were added
for V2, e.g. "blurred + resized + recompressed" in one shot, to better
match real redistribution). Restricting V2's own results to just the
15 conditions V0/V1 were scored on gives Avg Robust AUC = 0.9574, Avg
Robust Drop = 0.0128, Final Score = 0.9638 -- still the best of the
four, by a similar margin, so the ranking above holds either way, but
the exact V2 numbers in the table are not on the identical grid as
V0/V1's. We did not rerun V0/V1 against the 19-condition grid: a full
V2 cache+train+evaluate cycle already cost ~12 hours, and V0/V1 don't
need re-training, only re-evaluation, but re-running that evaluation
wasn't judged worth the remaining time versus finishing the
deliverable scripts.

**Held-out-generator generalization -- the most on-brief finding we
have.** `wildfake_dalle`'s "dalle" generator never appears in training
(by design, so it's a genuine unseen-generator test, unlike
`sid_set`'s generator label, which does appear in both train and
validation). Across every condition, AUC restricted to
coco_val2017-vs-wildfake_dalle rows runs noticeably below the pooled
AUC (e.g. 0.9559 vs. 0.9702 on clean for V2) -- a consistent ~1.5-2
point generalization gap. `error_analysis.py`'s output makes this
concrete: every single featured misclassification across the
`clean`/`blur_sigma2.0`/`resize_0.25` conditions was on a
`wildfake_dalle` image, and the same handful of specific images are
hard regardless of which of those three conditions they're in --
meaning the failure mode is "never saw this generator's fingerprint,"
not "this particular transform broke the model."

### Ablation: frequency gate + failure-condition reweighting

Two changes, evaluated together as one retrain, both aimed at a
specific, measured problem rather than a general robustness push:
`error_analysis.py` on the baseline V2 checkpoint showed real
`coco_val2017` images being misclassified as fake concentrated in
`blur_sigma2.0` and `resize_0.25` -- conditions that push residual
energy low enough that the frequency stream, which was trained mostly
on higher-energy inputs, starts producing an unreliable signal for
real photos specifically.

  - **Gate** (`FusionHead(use_freq_gate=True)`, opt-in): a tiny
    `Linear(1,8) -> ReLU -> Linear(8,1) -> Sigmoid` on top of the
    already-computed residual-energy scalar, multiplying the frequency
    embedding by that gate before fusion -- low residual energy ->
    gate closer to 0 -> the fusion head leans on the spatial stream
    instead of a frequency signal we have reason to distrust at that
    energy level.
  - **Reweighting** (`augmentations.FAILURE_UPWEIGHTED_CONDITIONS`,
    on by default): `sample_condition_names()` now samples
    `blur_sigma2.0` and `resize_0.25` roughly 2x as often during V2's
    augmented-embedding caching, so the model sees more training
    examples in the exact regime it was failing on.

Trained with `train.py --stage v2 --freq-mode fft --use-freq-gate`,
evaluated and calibrated the same way as the baseline, written to
`checkpoints/v2_augmented_fft_gated/` and
`results/v2_augmented_fft_gated.json` so neither overwrites the
baseline's checkpoint or results (see `evaluate.py --result-name` and
`train.py`'s `_gated` directory suffix).

**Result: the targeted fix worked, cleanly, and the tradeoff we
predicted before running it also showed up, more broadly than just the
two targeted conditions.**

| condition | FPR (old -> new) | pooled AUC (old -> new) | accuracy (old -> new) |
|---|:---:|:---:|:---:|
| `blur_sigma2.0` | 12.3% -> 7.9% | 0.9065 -> 0.9262 | 83.8% -> 84.4% |
| `resize_0.25`   | 13.8% -> 7.7% | 0.9072 -> 0.9281 | 83.8% -> 85.8% |

Almost the entire FPR drop traces to `coco_val2017` specifically
(blur: 17.0% -> 10.3%; resize: 18.7% -> 10.3% -- `sid_set`'s FPR under
the same conditions barely moved, it was never the source of the
problem). The AUC improvement on both conditions confirms this is a
genuine ranking improvement from the gate, not a threshold artifact.

The cost: FNR rose on both targeted conditions as expected (blur:
20.3% -> 23.4%; resize: 18.7% -> 21.0%), driven by `wildfake_dalle`
FNR climbing further (already the hardest source). More importantly,
the cost isn't confined to the two targeted conditions -- global mean
FNR rose 19.03% -> 19.71%, with `stack_severe` (the hardest compound
condition) taking the single largest hit at +5.9pp. Global mean FPR
looks flat (3.92% -> 3.91%) only because gains and losses cancel out:
`clean`-condition FPR on `coco_val2017` nearly doubled (2.3% -> 4.3%),
and FPR also rose on `jpeg_q30`, `noise_sigma0.05`, `noise_sigma0.1`,
and `color_jitter`. The residual-energy gate is reacting to more than
just blur/resize degradation -- JPEG compression, noise, and color
jitter all move residual energy too, so the gate suppresses the
frequency stream in some cases it wasn't meant to. The
held-out-generator AUC on `clean` -- this project's single most
important documented number -- also dipped slightly, 0.9559 -> 0.9527,
in exactly the direction the gate's mechanism predicts: suppressing
the frequency stream at low energy marginally weakens the one signal
`wildfake_dalle`'s hardest fakes rely on being caught by.

**Why this stays a documented ablation, not the submission checkpoint
for V2.** It's a real, verified improvement on the specific problem it
targeted, but a genuine tradeoff overall -- broader (if individually
small) FNR cost, a small held-out-generator AUC regression -- not a
strict win. Reported here alongside the baseline so both numbers are
part of the submission rather than only the more flattering one; use
`--checkpoint checkpoints/v2_augmented_fft_gated/model.pt` with
`infer.py`/`explain.py`/`error_analysis.py` to reproduce or inspect it
directly (they all detect the gated architecture automatically via
`checkpoints/v2_augmented_fft_gated/architecture.json`, no code
changes needed).

## Setup

    uv sync

(Requires network access to fetch torch and friends -- run this from a
regular terminal if it fails in a sandboxed shell.)

## Data

`data/` is gitignored (it's tens of thousands of images) -- none of it
ships in this repo. `src/data/datasets.py` scans four fixed folder
layouts and silently skips whichever ones aren't present yet (see its
module docstring), so you can rebuild the manifest incrementally as
you add each source. Expected layout, exactly what `datasets.py`'s
`scan_*()` functions look for:

    data/
      train/
        cifake/
          train/REAL/*.jpg   train/FAKE/*.jpg
          test/REAL/*.jpg    test/FAKE/*.jpg
        sid_set/
          real/*.jpg  fake/*.jpg  manifest.csv
      validation_demo/
        coco_val2017/*.jpg          (5,000 real images)
        dalle_advanced/**/*.{jpg,png}   (8,844 AI-generated images)

How to populate each one:

  - **CIFAKE** (real photos + Stable-Diffusion fakes, used for
    training): download from Kaggle --
    https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
    -- and extract it so `data/train/cifake/{train,test}/{REAL,FAKE}/`
    match the layout above (that's the zip's own folder structure,
    unchanged).
  - **SID_Set** (real + fully-AI-generated images, used for training,
    plus a held-out validation slice): run
    `scripts/download_sid_set.py` (`pip install datasets pillow`
    first) -- it pulls directly from
    https://huggingface.co/datasets/saberzl/SID_Set and writes
    `data/train/sid_set/` in the layout above automatically. See that
    script's own docstring for the subset sizes and label mapping.
  - **`validation_demo/coco_val2017` + `validation_demo/dalle_advanced`**
    (real COCO photos + WildFake's DALL-E-Advanced subset): this is
    the competition's own reference/demonstration benchmark from
    brief section 5.4 -- our counts (5,000 / 8,844) match its stated
    4,998 / 8,843 almost exactly, so this is that exact set, not a
    resample. It's sourced from WildFake
    (https://modelscope.cn/datasets/hy2628982280/WildFake/summary (specifically DALL-E's Advanced folder) --
    use the translation button before browsing it, per the brief) and
    the standard COCO val2017 images. This is the ONLY data source used for `validation_demo`
    (aside from SID_Set's own validation rows, folded in
    automatically -- see `datasets.py`); it is never trained on
    (`build_manifest()` asserts this).

Once the folders above exist, `uv run python src/data/datasets.py`
builds `data/manifest.csv`, and `scripts/build_eval_grid.py` builds
`data/eval_transformed/` + `data/eval_manifest.csv` (the 19-condition
robustness grid, sampled from `validation_demo`) -- both idempotent,
safe to rerun after adding a new data source.

`cache/`, `checkpoints/`, and `outputs/` are also gitignored, but
don't need manual setup -- they're generated by
`cache_embeddings.py`, `train.py`/`calibrate.py`, and
`infer.py`/`explain.py` respectively, all covered in "Reproducing
results" below.

## Reproducing results

From the repo root, after `uv sync` and getting the raw data in place
per "Data" above:

    # 1. Build the training manifest and the robustness eval grid.
    uv run python src/data/datasets.py
    uv run python scripts/build_eval_grid.py

    # 2. Cache CLIP embeddings (clean + augmented variants for V2).
    uv run python src/cache_embeddings.py --variant clean
    uv run python src/cache_embeddings.py --variant augmented

    # 3. Train each stage (V0 and V1 are quick; V2 is the ~12-hour one).
    uv run python src/train.py --stage v0
    uv run python src/train.py --stage v1                  # "srm" mode (default)
    uv run python src/train.py --stage v1 --freq-mode fft   # "fft" ablation
    uv run python src/train.py --stage v2 --num-workers 6   # tune --num-workers to your CPU

    # 4. Evaluate each stage against the robustness grid.
    uv run python src/evaluate.py --stage v0
    uv run python src/evaluate.py --stage v1
    uv run python src/evaluate.py --stage v1 --freq-mode fft
    uv run python src/evaluate.py --stage v2 --dump-predictions   # --dump-predictions feeds error_analysis.py

    # 5. Calibrate V2's raw scores into real probabilities.
    uv run python src/calibrate.py --stage v2

    # 5b. Optional: the frequency-gate + reweighting ablation (see
    #     "Ablation: frequency gate + failure-condition reweighting"
    #     above) -- writes to a separate checkpoint/results name, does
    #     NOT overwrite step 3-5's V2 output.
    uv run python src/train.py --stage v2 --freq-mode fft --use-freq-gate
    uv run python src/evaluate.py --stage v2 --freq-mode fft \
        --checkpoint checkpoints/v2_augmented_fft_gated/model.pt \
        --result-name v2_augmented_fft_gated --dump-predictions
    uv run python src/calibrate.py --stage v2 --checkpoint checkpoints/v2_augmented_fft_gated/model.pt

    # 6. The actual deliverable: score any folder of images.
    uv run python src/infer.py --input path/to/some/images

    # 7. Explainability + error analysis.
    uv run python src/explain.py --mode gradcam --image path/to/one.jpg
    uv run python src/explain.py --mode spectrum
    uv run python src/error_analysis.py

Step 3's V2 training is the expensive one -- see its own `--help` and
`train.py`'s module docstring for `--num-workers` guidance if it's
running on limited CPU/RAM.

## Limitations & what we'd improve with more time

**V3 (consistency loss) is the biggest one, and it's a scope decision,
not an accident.** Our own roadmap calls it "our main proposed
method" -- explicitly penalizing a model for disagreeing with itself
on a clean vs. transformed view of the same image, on top of
augmentation alone. V2's full pipeline already cost ~12 hours
end-to-end, and V3's training step needs two forward passes per
example instead of one, so it would cost more per epoch, not less.
Architecturally nothing else changes for V3 -- `Detector`,
`SpatialStream`, `FrequencyStream`, and `FusionHead` are all reused
unchanged, only the training loss/procedure differs -- so every script
built this hackathon (`infer.py`, `explain.py`, `calibrate.py`,
`error_analysis.py`) would work against a V3 checkpoint with no code
changes, just a different `--checkpoint` path. Given more time, V3 is
the clear next step, not a redesign.

**Generalization to unseen generators is the real weak point, not
robustness to transforms.** See the Robustness Evaluation Summary
above -- the gap between pooled AUC and held-out-generator AUC is
consistent and larger than most of the transform-induced drops, and
every misclassification `error_analysis.py` surfaced was on the one
generator family (`wildfake_dalle`) never seen in training. More
generator families in training data (or a generator-invariant training
objective) would likely move this number more than further robustness
engineering would. We have a direct data point for this now: the
frequency-gate ablation (see above) fixed a real robustness problem
(blur/resize FPR on real images) and still slightly *regressed* the
held-out-generator AUC (0.9559 -> 0.9527) -- further evidence that
transform-robustness and generator-generalization are separate axes
here, and that we picked the harder, more valuable one correctly as
"the real weak point" rather than assuming robustness work would help
both.

**A possible frequency-domain shortcut we did not rule out.** CIFAKE
pairs real CIFAR-10 photos against Stable-Diffusion-generated
counterparts; if the two sides of that pair systematically differ in
native compression or resolution characteristics independent of
"AI-generated-ness," a detector can learn that confound instead of a
genuine forensic signal (this is a documented failure mode in the
AIGC-detection literature, not specific to this project). We didn't
run the control experiment that would confirm or rule this out (e.g.
scoring real-vs-fake pairs matched for compression from a source not
used in training) -- worth doing before trusting the frequency
stream's contribution too far.

**`sid_set`'s generator label appears in both train and validation.**
Unlike `wildfake_dalle`, `sid_set_mixed` shows up in both splits, so
any of its rows in the eval grid measure "held-out samples," not
"held-out generator family" -- `evaluate.py` already excludes it from
the unseen-generator view for exactly this reason, but it's worth
being explicit that this limits how much of the eval grid is a true
generalization test.

**Calibration reduced overconfidence penalty a lot, but not binned
calibration error by much.** Fitting temperature scaling on V2 dropped
NLL substantially (1.12 -> 0.38) but only nudged ECE (0.140 -> 0.130).
That's an honest signal that the model's confidence is still not
well-calibrated in a bucketed sense even after fitting a single global
temperature -- a per-condition or per-source temperature, or a
non-parametric calibration method, would likely do better, at the cost
of needing more calibration data per slice.

**Grad-CAM on the spatial stream needs a layer choice, and there's no
single right answer.** Hooking the LAST transformer block (the
default) can produce a nearly-flat, uninformative heatmap for some
images -- by that point self-attention has mixed every patch token
with global context, which is a known limitation of Grad-CAM on ViTs
at the final layer. `explain.py --layer <n>` lets you pick an earlier
block instead (a middle layer, e.g. 6 of 12 for ViT-B/32, gave a much
more localized and interpretable result in our own testing, landing on
the monitor screen and hands in one example -- both classic
AI-generation tells). This is left as something to compare a few
layers of, not something the script picks automatically, since later
layers are more relevant to the actual decision while earlier layers
are more spatially localized, and there's a real trade-off there.

**No true production deployment considerations.** Per the brief's own
scope, this is a hackathon-scale prototype: no throughput/latency
benchmarking, no model compression, no monitoring for distribution
drift once deployed. `<2B` parameters is comfortably satisfied (CLIP
ViT-B/32 is roughly 150M parameters; the frequency CNN and fusion head
together are under 1M), so there's headroom to consider a larger
backbone if a production setting justified the added cost.

## Error analysis

`src/error_analysis.py` reads a checkpoint's dumped predictions
(`evaluate.py --dump-predictions`) and pulls specific misclassified
examples, splitting them into **shared blind spots** (the spatial-only
V0 model also got it wrong -- the frequency stream/fusion head had
nothing better to fall back on) vs. **fusion-induced regressions** (V0
was right, but fusing in the frequency stream flipped it wrong) --
full output in `results/v2_error_analysis.csv` /
`results/v2_error_analysis.md`. Headline finding: on the baseline V2
checkpoint's `clean`, `blur_sigma2.0`, and `resize_0.25` conditions,
every featured misclassification was a shared blind spot, 0 were
fusion-induced -- the frequency stream isn't the thing making errors
worse on its own hardest cases, it just isn't rescuing V0's hardest
cases either. All of them were `wildfake_dalle` images (see
"Held-out-generator generalization" above), consistent with a
generator-fingerprint gap rather than a fusion bug.

The frequency-gate ablation's own error pattern is the FPR/FNR-by-source
breakdown in "Ablation: frequency gate + failure-condition reweighting"
above (real `coco_val2017` images wrongly flagged as fake under
blur/resize, and why fixing that cost accuracy on `wildfake_dalle`
fakes and on the hardest compound condition, `stack_severe`) -- that
table is effectively the same FP/FN trade-off discussion for the
gated checkpoint, done at the aggregate level rather than
per-example, and is what actually drove the choice in "Submission
model" above.

## Team contributions

Ng Yin Xuan Sarah, Tay Wen Xin, Ysabelle Wong Sze Han, Zhang Qiyun
