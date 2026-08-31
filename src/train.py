"""
Training entry point.

V0 -- spatial stream only: train a small head directly on the CACHED
clean spatial embeddings from cache_embeddings.py, with plain binary
cross-entropy. This answers the roadmap's V0 question -- "is CLIP + a
small head already separating real from fake at all" -- with the
smallest thing that could work, before the frequency stream or fusion
head exist. (Done -- clean AUC 0.953, see results/v0_spatial_only.json.)

V1 -- + frequency stream: reuses those SAME cached spatial embeddings
(still frozen, still the CPU bottleneck worth avoiding recomputation
of) and pairs each one with a freshly-computed frequency-stream input
for the same image, then trains models.frequency_stream.FrequencyStream
(from scratch) and models.fusion.FusionHead jointly through a shared
optimizer. Answered the roadmap's V1 question -- "does the forensic
branch actually add anything, or is the spatial stream carrying all the
signal on its own" -- yes: final score 0.939 -> 0.959, "fft" mode beats
"srm" on 13/15 conditions (see results/v1_fusion_*.json).

V2 -- + transform-aware augmentation: trains on a MIX of clean and
augmented cached embeddings (cache_embeddings.py --variant augmented),
so the fusion head and frequency CNN actually see transformed images
during training, not just at eval time. This is also where V1's
checkpoint-selection blind spot gets fixed: V1's val set was clean-only
(nothing in V1 ever validated against transformed data), so "lowest
val_loss" could only ever reflect clean-data generalization, not
robustness -- and empirically it didn't even reliably track the
robustness-grid AUC (see the v1_fusion_srm before/after-fix comparison
in results/). V2's val set includes both clean AND augmented rows (see
train_v2 / _BestCheckpointTracker), so "best epoch" now means "best on
a validation signal that actually includes transformed images" for the
first time.

Objective for V3 (our main proposed method -- not implemented yet,
see the architecture doc's roadmap for V2 first):

    L = L_classification(y_hat, y) + L_classification(y_hat_t, y)
        + lambda * L_consistency(y_hat, y_hat_t)

where y_hat = Detector.predict(image), y_hat_t = Detector.predict(transform(image))
using the SAME weights, and L_consistency penalizes the two predictions
for disagreeing (e.g. MSE or symmetric KL on the probabilities). lambda
is config.CONSISTENCY_LOSS_WEIGHT -- log L_classification and
L_consistency separately during training, not just their sum: a
consistency term weighted too heavily can collapse both predictions
toward an uninformative middle value instead of actually improving
robustness.

Roadmap (build and evaluate in this order; keep every checkpoint and
its metrics under results/, never overwrite):
    V0  spatial stream only                    (done)
    V1  + frequency stream                     (done)
    V2  + transform-aware augmentation         (this file's train_v2, below)
    V3  + consistency loss                     (our main method)
    V4  optional: learned gating / partial backbone fine-tuning

Run with:  uv run python src/cache_embeddings.py --split train                       (once)
           uv run python src/cache_embeddings.py --split train --variant augmented   (once, for V2)
           uv run python src/train.py --stage v0
           uv run python src/train.py --stage v1
           uv run python src/train.py --stage v1 --freq-mode fft   (the srm-vs-fft ablation)
           uv run python src/train.py --stage v2 --limit 2000      (smoke test first -- V2's combined
                                                                      clean+augmented set is ~4x V1's size)
           uv run python src/train.py --stage v2
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import CHECKPOINT_DIR, EMBEDDING_CACHE_DIR, FREQUENCY_MODE, SPATIAL_EMBED_DIM
from models.frequency_stream import FrequencyStream
from models.fusion import FusionHead, save_architecture_metadata
from transforms.augmentations import apply_condition
from transforms.preprocessing import prepare_frequency_input
from transforms.preprocessing import residual_energy as compute_residual_energy


class V0Head(nn.Module):
    """Minimal classifier head for V0. See module docstring for why
    this isn't models.fusion.FusionHead."""

    def __init__(self, embed_dim: int = SPATIAL_EMBED_DIM, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits, shape (N,)


def load_cached(split: str, cache_dir: str = EMBEDDING_CACHE_DIR):
    cache_dir = Path(cache_dir)
    npy_path = cache_dir / f"{split}_clean.npy"
    index_path = cache_dir / f"{split}_clean_index.csv"
    if not npy_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"{npy_path} / {index_path} not found -- run "
            f"`uv run python src/cache_embeddings.py --split {split}` first."
        )
    embeddings = np.load(npy_path)
    with open(index_path, newline="") as f:
        index_rows = list(csv.DictReader(f))
    assert len(index_rows) == len(embeddings), (
        f"cache index ({len(index_rows)} rows) and embedding array ({len(embeddings)} rows) "
        f"disagree in length -- rerun cache_embeddings.py for split={split!r}"
    )
    return embeddings, index_rows


def load_cached_augmented(split: str, cache_dir: str = EMBEDDING_CACHE_DIR):
    cache_dir = Path(cache_dir)
    npy_path = cache_dir / f"{split}_augmented.npy"
    index_path = cache_dir / f"{split}_augmented_index.csv"
    if not npy_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"{npy_path} / {index_path} not found -- run "
            f"`uv run python src/cache_embeddings.py --split {split} --variant augmented` first."
        )
    embeddings = np.load(npy_path)
    with open(index_path, newline="") as f:
        index_rows = list(csv.DictReader(f))
    assert len(index_rows) == len(embeddings), (
        f"cache index ({len(index_rows)} rows) and embedding array ({len(embeddings)} rows) "
        f"disagree in length -- rerun `cache_embeddings.py --variant augmented` for split={split!r}"
    )
    return embeddings, index_rows


def _drop_nan_rows(embeddings, index_rows):
    """Shared by train_v0/v1/v2: cache_embeddings.py writes an all-NaN
    row (rather than skipping) for any image/condition it couldn't
    process, to keep the array and index aligned -- see that module's
    docstring. Every training path needs to filter those out before use."""
    valid_mask = ~np.isnan(embeddings).any(axis=1)
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        print(f"dropping {n_dropped} unreadable row(s) flagged during caching")
    embeddings = embeddings[valid_mask]
    index_rows = [r for r, keep in zip(index_rows, valid_mask) if keep]
    return embeddings, index_rows


class _BestCheckpointTracker:
    """Tracks the best-val-loss epoch's weights across training,
    instead of just keeping whatever the last epoch happened to be.
    Shared by train_v1 and train_v2 so this logic can't drift between
    the two -- it matters more for V2 than V1 did, since V2's larger,
    more varied training set (clean + augmented) has correspondingly
    more room for a late epoch to drift worse on some slice of the data
    even while looking fine in aggregate."""

    def __init__(self):
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_state = None

    def update(self, val_loss: float, epoch: int, modules: dict) -> bool:
        """modules: {name: nn.Module} -- every module whose state_dict()
        gets snapshotted when this epoch is the new best. Returns
        whether this epoch WAS the new best (for a caller's own
        logging)."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_state = {
                name: {k: v.detach().clone() for k, v in module.state_dict().items()}
                for name, module in modules.items()
            }
            return True
        return False


def train_v0(epochs: int = 10, lr: float = 1e-3, batch_size: int = 256, val_fraction: float = 0.1, seed: int = 0):
    embeddings, index_rows = load_cached("train")
    embeddings, index_rows = _drop_nan_rows(embeddings, index_rows)

    labels = np.array([int(r["label"]) for r in index_rows], dtype=np.float32)
    print(f"training on {len(embeddings)} cached embeddings "
          f"(real={int((labels == 0).sum())}, fake={int((labels == 1).sum())})")

    rng = np.random.default_rng(seed)
    n = len(embeddings)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_train = torch.from_numpy(embeddings[train_idx])
    y_train = torch.from_numpy(labels[train_idx])
    X_val = torch.from_numpy(embeddings[val_idx])
    y_val = torch.from_numpy(labels[val_idx])

    head = V0Head()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n_train = len(X_train)
    for epoch in range(1, epochs + 1):
        head.train()
        epoch_perm = torch.randperm(n_train)
        total_loss = 0.0
        for start in range(0, n_train, batch_size):
            batch_idx = epoch_perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = head(X_train[batch_idx])
            loss = loss_fn(logits, y_train[batch_idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        train_loss = total_loss / n_train

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val)
            val_loss = loss_fn(val_logits, y_val).item()
            val_acc = ((val_logits > 0).float() == y_val).float().mean().item()
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    ckpt_dir = Path(CHECKPOINT_DIR) / "v0_spatial_only"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "head.pt"
    torch.save(head.state_dict(), ckpt_path)
    print(f"\nSaved checkpoint -> {ckpt_path}")
    return head


class _FrequencyTrainDataset(Dataset):
    """Pairs a row's already-cached (frozen) spatial embedding with a
    freshly-computed frequency-stream input for the SAME image. Used by
    train_v1 (every row is the clean image). train_v2 uses the more
    general _FrequencyV2Dataset below instead, which also knows how to
    reapply a non-clean condition before frequency preprocessing.

    Unlike the spatial embedding, the frequency-stream input is NOT
    cached to disk: cache_embeddings.py's caching strategy exists
    specifically because re-running the *frozen* CLIP backbone every
    epoch is the actual CPU bottleneck (see the architecture doc's
    "Compute-aware order of operations" note). The frequency-stream
    preprocessing here (a 5x5 convolution + a resize, in numpy/scipy)
    is far cheaper per-call than a ViT forward pass, and -- more to the
    point -- its CNN is what's being TRAINED here: caching only the raw
    residual/spectrum map would still require a full forward + backward
    pass through the CNN every step regardless, so caching would only
    save the (already cheap) preprocessing step, not the (already the
    actual cost center) CNN step. Recomputing it per-epoch, spread
    across DataLoader worker processes so it overlaps with the main
    process's forward/backward pass, keeps this simple.

    Calls transforms.preprocessing directly rather than going through
    models.frequency_stream.FrequencyStream.prepare(): that avoids
    constructing a FrequencyStream (and its CNN) inside every
    DataLoader worker process, which would be needless overhead here
    since the worker's only job is producing the pre-CNN input tensor.
    """

    def __init__(self, spatial_embeddings, index_rows, freq_mode: str = None):
        assert len(spatial_embeddings) == len(index_rows), (
            f"{len(spatial_embeddings)} embeddings vs {len(index_rows)} index rows -- "
            f"these must stay aligned 1:1"
        )
        # Shared memory, not a private numpy array per DataLoader worker.
        # macOS spawns DataLoader workers as separate processes rather
        # than forking them, so a plain numpy array attribute here would
        # get pickled and fully duplicated in every worker process --
        # for a training split this size that's real memory (measured,
        # not theoretical: it's what made a modest --num-workers bump
        # exhaust available RAM during V2 sizing). Moving the array into
        # a torch shared-memory tensor ONCE here, in the main process,
        # before any worker is spawned, means every worker maps the SAME
        # underlying buffer instead of copying it. This is safe because
        # nothing ever writes to self.embeddings after this line --
        # every worker only reads row i -- so it changes memory layout
        # only: it can't change which row a given index returns, what
        # order rows are read in, or any computed value. Not something
        # that could show up as an accuracy difference.
        self.embeddings = torch.from_numpy(spatial_embeddings).share_memory_()
        self.rows = index_rows
        self.freq_mode = freq_mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        # Deliberately NOT wrapped in a try/except that falls back to a
        # NaN/zero row the way cache_embeddings.py does: every row here
        # already passed that exact read-and-decode step once, when it
        # was cached (rows that failed then were already dropped by
        # _drop_nan_rows before this Dataset was ever built). A failure
        # here now means something changed on disk since caching --
        # worth a loud crash pointing at the path, not a silent skip.
        img = Image.open(row["path"]).convert("RGB")
        freq_map = prepare_frequency_input(img, mode=self.freq_mode)
        energy = compute_residual_energy(freq_map)

        spatial_embedding = self.embeddings[i]  # already a tensor -- see __init__'s shared-memory note
        freq_tensor = torch.from_numpy(freq_map).unsqueeze(0)  # (1, H, W)
        energy = torch.tensor(energy, dtype=torch.float32)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return spatial_embedding, freq_tensor, energy, label


def train_v1(
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_fraction: float = 0.1,
    seed: int = 0,
    num_workers: int = 4,
    freq_mode: str = None,
):
    embeddings, index_rows = load_cached("train")
    embeddings, index_rows = _drop_nan_rows(embeddings, index_rows)

    labels_preview = np.array([int(r["label"]) for r in index_rows])
    print(f"training on {len(embeddings)} cached embeddings "
          f"(real={int((labels_preview == 0).sum())}, fake={int((labels_preview == 1).sum())})")

    # Same seed + same underlying row order as train_v0's default call
    # -> the same train/val split, so V0's and V1's val_acc are directly
    # comparable (the real head-to-head is evaluate.py's held-out
    # robustness grid either way, but this costs nothing extra).
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_rows = [index_rows[i] for i in train_idx]
    val_rows = [index_rows[i] for i in val_idx]
    train_ds = _FrequencyTrainDataset(embeddings[train_idx], train_rows, freq_mode=freq_mode)
    val_ds = _FrequencyTrainDataset(embeddings[val_idx], val_rows, freq_mode=freq_mode)

    # persistent_workers keeps the same worker processes alive across
    # every epoch's DataLoader iteration, instead of spawning a fresh
    # pool every single time a `for ... in loader:` loop starts (which
    # is the default -- the DataLoader OBJECT being created once outside
    # the epoch loop does NOT by itself avoid this; PyTorch tears down
    # and respawns workers on every new __iter__() call unless this flag
    # is set). On macOS specifically, each spawn re-imports this whole
    # module (including `import torch`) in a brand new process, so with
    # num_workers>0 that's num_workers respawns x 2 loaders x every
    # epoch -- for a 10-epoch run that repeated spawn overhead is pure
    # waste, since nothing about the workers needs to change epoch to
    # epoch. Guarded on num_workers>0 because PyTorch raises if this is
    # set with num_workers=0 (nothing to persist).
    persistent_workers = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=persistent_workers)

    resolved_mode = freq_mode or FREQUENCY_MODE
    print(f"Building frequency stream (mode={resolved_mode}, trained from scratch) + fusion head...")
    freq_stream = FrequencyStream(freeze=False, mode=freq_mode)
    fusion = FusionHead()

    optimizer = torch.optim.Adam(
        list(freq_stream.model.parameters()) + list(fusion.parameters()), lr=lr
    )
    loss_fn = nn.BCEWithLogitsLoss()

    def run_epoch(loader, train: bool, epoch: int = None):
        freq_stream.model.train(train)
        fusion.train(train)
        total_loss, n_seen, n_correct = 0.0, 0, 0
        grad_context = torch.enable_grad() if train else torch.no_grad()
        # tqdm here is purely visibility -- it reports progress through
        # an epoch's batches, it does not change what data is read, what
        # order it's read in, or any computed value. Without this, a
        # slow epoch prints nothing at all until it's fully done (see
        # the print() after both run_epoch calls below), which made it
        # impossible to tell a genuinely slow epoch apart from a hung
        # process.
        desc = f"epoch {epoch}/{epochs} {'train' if train else 'val'}" if epoch else ("train" if train else "val")
        with grad_context:
            for spatial_emb, freq_tensor, energy, label in tqdm(loader, desc=desc, leave=False):
                if train:
                    optimizer.zero_grad()
                freq_emb = freq_stream.encode(freq_tensor)
                logits = fusion(spatial_emb, freq_emb, energy)
                loss = loss_fn(logits, label)
                if train:
                    loss.backward()
                    optimizer.step()
                bs = label.shape[0]
                total_loss += loss.item() * bs
                n_correct += ((logits > 0).float() == label).sum().item()
                n_seen += bs
        return total_loss / n_seen, n_correct / n_seen

    tracker = _BestCheckpointTracker()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True, epoch=epoch)
        val_loss, val_acc = run_epoch(val_loader, train=False, epoch=epoch)
        is_best = tracker.update(val_loss, epoch, {"frequency_cnn": freq_stream.model, "fusion_head": fusion})
        marker = "  (best so far, saved)" if is_best else ""
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{marker}")

    ckpt_dir = Path(CHECKPOINT_DIR) / f"v1_fusion_{resolved_mode}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.pt"
    torch.save(
        {
            "frequency_cnn": tracker.best_state["frequency_cnn"],
            "fusion_head": tracker.best_state["fusion_head"],
            "freq_mode": resolved_mode,
            "best_epoch": tracker.best_epoch,
            "best_val_loss": tracker.best_val_loss,
            "trained_epochs": epochs,
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint from epoch {tracker.best_epoch}/{epochs} "
          f"(val_loss={tracker.best_val_loss:.4f}) -> {ckpt_path}")
    freq_stream.model.load_state_dict(tracker.best_state["frequency_cnn"])
    fusion.load_state_dict(tracker.best_state["fusion_head"])
    return freq_stream, fusion


class _FrequencyV2Dataset(Dataset):
    """Like _FrequencyTrainDataset, but each row also carries a
    `condition` -- "clean" for a V0/V1-style cached row, or one of
    transforms.augmentations.ALL_CONDITIONS' names for a row from
    cache_embeddings.py's --variant augmented cache. That condition is
    reapplied to the raw image before frequency preprocessing runs: the
    frequency branch has to see the SAME transformed image the cached
    spatial embedding was computed from, or the two branches would be
    voting on two different pictures for what's supposed to be one
    training example.

    Also returns the condition string itself (not just the tensors) so
    train_v2's run_epoch can report clean-vs-augmented validation
    accuracy separately -- the actual "does the clean-vs-transformed
    accuracy gap shrink" number the V2 roadmap entry asks about, not
    just one pooled val_loss that could hide it either way.
    """

    def __init__(self, spatial_embeddings, index_rows, freq_mode: str = None):
        assert len(spatial_embeddings) == len(index_rows), (
            f"{len(spatial_embeddings)} embeddings vs {len(index_rows)} index rows -- "
            f"these must stay aligned 1:1"
        )
        # Shared memory, not a private numpy array per DataLoader worker
        # -- see _FrequencyTrainDataset's __init__ for why this matters
        # (macOS spawns workers as separate processes, so a plain numpy
        # array here would otherwise be duplicated in full per worker),
        # and why it's safe (nothing ever writes to this array after
        # construction, so no row's value, order, or any downstream
        # computed number changes -- only where the bytes physically
        # live in memory does). This matters more for V2 than V1: the
        # combined clean+augmented split is the larger of the two.
        self.embeddings = torch.from_numpy(spatial_embeddings).share_memory_()
        self.rows = index_rows
        self.freq_mode = freq_mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        condition = row["condition"]
        img = Image.open(row["path"]).convert("RGB")
        if condition != "clean":
            img = apply_condition(img, condition)
        freq_map = prepare_frequency_input(img, mode=self.freq_mode)
        energy = compute_residual_energy(freq_map)

        spatial_embedding = self.embeddings[i]  # already a tensor -- see __init__'s shared-memory note
        freq_tensor = torch.from_numpy(freq_map).unsqueeze(0)
        energy = torch.tensor(energy, dtype=torch.float32)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return spatial_embedding, freq_tensor, energy, label, condition


def train_v2(
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_fraction: float = 0.1,
    seed: int = 0,
    num_workers: int = 4,
    freq_mode: str = None,
    limit: int = None,
    use_freq_gate: bool = False,
):
    clean_embeddings, clean_rows = load_cached("train")
    clean_embeddings, clean_rows = _drop_nan_rows(clean_embeddings, clean_rows)
    for r in clean_rows:
        r["condition"] = "clean"

    aug_embeddings, aug_rows = load_cached_augmented("train")
    aug_embeddings, aug_rows = _drop_nan_rows(aug_embeddings, aug_rows)

    all_embeddings = np.concatenate([clean_embeddings, aug_embeddings], axis=0)
    all_rows = clean_rows + aug_rows
    del clean_embeddings, aug_embeddings  # ~1GB combined at the full dataset size -- free the now-redundant copies

    unique_paths = sorted(set(r["path"] for r in all_rows))
    print(f"loaded {len(clean_rows)} clean + {len(aug_rows)} augmented rows "
          f"({len(all_rows)} total, {len(unique_paths)} unique images)")

    if limit:
        # Restricted at the IMAGE level, not the row level, and BEFORE
        # the train/val split below -- so a smoke test still exercises
        # the same "does an image's clean row and its augmented rows
        # stay together" logic the full run relies on, just on fewer
        # images. (This still pays the cost of loading the full cached
        # arrays into memory first -- see module docstring's caveat on
        # V2's larger memory footprint -- but the load itself is a
        # one-time, bounded cost; --limit's actual purpose is bounding
        # the O(epochs x rows) training loop below, which this does.)
        #
        # Sampled from images that actually HAVE augmented rows cached
        # (aug_rows), not from every image in the manifest. Sampling
        # from all unique_paths here used to mean a smoke test's
        # `--limit N` and cache_embeddings.py's own `--limit M` (used to
        # bound the augmented cache itself) landed on two DIFFERENT
        # random N/M-image subsets, even with the same seed -- they
        # sample over different domains (this used the full sorted list
        # of all 123K image paths; cache_embeddings.py's
        # `_shuffled_subset` shuffles the raw manifest row order and
        # takes a head slice), so the two only overlapped by chance
        # (~N*M/123000 images -- about 33 for N=M=2000, not 2000). The
        # result: a smoke test's own "augmented" rows were almost
        # entirely for images outside its own clean sample, so the
        # dataset it actually trained and validated on had barely any
        # augmented rows left after the path filter below, and the
        # clean-vs-augmented val gap printed each epoch was computed
        # from a handful of leftover rows instead of a real augmented
        # sample. Restricting the candidate pool to aug_rows' own images
        # first guarantees every sampled image has real augmented
        # variants available, regardless of what --limit (if any)
        # cache_embeddings.py was run with.
        augmented_paths = sorted(set(r["path"] for r in aug_rows))
        if not augmented_paths:
            raise RuntimeError(
                "--limit was given but the augmented cache has zero rows to sample from -- "
                "run `cache_embeddings.py --split train --variant augmented` first."
            )
        n_images = min(limit, len(augmented_paths))
        if n_images < limit:
            print(f"--limit {limit} requested, but only {len(augmented_paths)} images have "
                  f"cached augmented rows -- using all {n_images} of them")
        chosen = set(random.Random(seed).sample(augmented_paths, n_images))
        keep_idx = [i for i, r in enumerate(all_rows) if r["path"] in chosen]
        all_embeddings = all_embeddings[keep_idx]
        all_rows = [all_rows[i] for i in keep_idx]
        unique_paths = sorted(chosen)
        print(f"--limit {limit}: restricted to {len(unique_paths)} images, {len(all_rows)} rows")

    # Split at the IMAGE level, not the row level: a source image's
    # clean embedding and all of its augmented variants must land on
    # the SAME side of the split. Splitting by row instead would let,
    # say, image X's clean embedding sit in train while X's blurred
    # variant sits in val -- that's not a held-out image, it's the same
    # image leaking across the split under a different condition name,
    # and it would make val_loss look better than genuine held-out
    # performance actually is. This is also exactly what makes V2's
    # validation trustworthy for checkpoint selection in a way V1's
    # (clean-only) validation wasn't -- see module docstring.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_paths))
    n_val_images = max(1, int(len(unique_paths) * val_fraction))
    val_paths = {unique_paths[i] for i in perm[:n_val_images]}

    train_idx = [i for i, r in enumerate(all_rows) if r["path"] not in val_paths]
    val_idx = [i for i, r in enumerate(all_rows) if r["path"] in val_paths]

    train_rows = [all_rows[i] for i in train_idx]
    val_rows = [all_rows[i] for i in val_idx]
    n_val_clean = sum(1 for r in val_rows if r["condition"] == "clean")
    print(f"train: {len(train_rows)} rows over {len(unique_paths) - len(val_paths)} images | "
          f"val: {len(val_rows)} rows over {len(val_paths)} images "
          f"({n_val_clean} clean, {len(val_rows) - n_val_clean} augmented)")

    train_ds = _FrequencyV2Dataset(all_embeddings[train_idx], train_rows, freq_mode=freq_mode)
    val_ds = _FrequencyV2Dataset(all_embeddings[val_idx], val_rows, freq_mode=freq_mode)
    # persistent_workers keeps the same worker processes alive across
    # every epoch's DataLoader iteration -- see train_v1's identical
    # comment above for why this matters (macOS respawns + re-imports
    # per __iter__() call by default, once per loader per epoch,
    # regardless of whether the DataLoader object itself is
    # reconstructed). Guarded on num_workers>0 since PyTorch raises if
    # this is set with num_workers=0.
    persistent_workers = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=persistent_workers)

    resolved_mode = freq_mode or FREQUENCY_MODE
    print(f"Building frequency stream (mode={resolved_mode}, trained from scratch) + fusion head "
          f"(use_freq_gate={use_freq_gate})...")
    freq_stream = FrequencyStream(freeze=False, mode=freq_mode)
    fusion = FusionHead(use_freq_gate=use_freq_gate)

    optimizer = torch.optim.Adam(
        list(freq_stream.model.parameters()) + list(fusion.parameters()), lr=lr
    )
    loss_fn = nn.BCEWithLogitsLoss()

    def run_epoch(loader, train: bool, epoch: int = None):
        freq_stream.model.train(train)
        fusion.train(train)
        total_loss, n_seen, n_correct = 0.0, 0, 0
        # Clean-vs-augmented breakdown -- only tallied during
        # validation (train doesn't need it, and it would just cost
        # extra work every step for no benefit): this is the actual
        # "clean-vs-transformed accuracy gap" the V2 roadmap entry asks
        # whether training through the transform grid shrinks.
        clean_correct, clean_seen, aug_correct, aug_seen = 0, 0, 0, 0
        grad_context = torch.enable_grad() if train else torch.no_grad()
        # tqdm is visibility only -- see train_v1's run_epoch for why it
        # was added (a slow epoch used to print nothing at all until it
        # fully finished, indistinguishable from a hung process). It
        # doesn't touch what data is read, its order, or any computed
        # value.
        desc = f"epoch {epoch}/{epochs} {'train' if train else 'val'}" if epoch else ("train" if train else "val")
        with grad_context:
            for spatial_emb, freq_tensor, energy, label, conditions in tqdm(loader, desc=desc, leave=False):
                if train:
                    optimizer.zero_grad()
                freq_emb = freq_stream.encode(freq_tensor)
                logits = fusion(spatial_emb, freq_emb, energy)
                loss = loss_fn(logits, label)
                if train:
                    loss.backward()
                    optimizer.step()
                bs = label.shape[0]
                correct = (logits > 0).float() == label
                total_loss += loss.item() * bs
                n_correct += correct.sum().item()
                n_seen += bs
                if not train:
                    for is_correct, condition in zip(correct.tolist(), conditions):
                        if condition == "clean":
                            clean_seen += 1
                            clean_correct += int(is_correct)
                        else:
                            aug_seen += 1
                            aug_correct += int(is_correct)
        result = {"loss": total_loss / n_seen, "acc": n_correct / n_seen}
        if not train:
            result["clean_acc"] = clean_correct / clean_seen if clean_seen else float("nan")
            result["aug_acc"] = aug_correct / aug_seen if aug_seen else float("nan")
        return result

    tracker = _BestCheckpointTracker()
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(train_loader, train=True, epoch=epoch)
        val_metrics = run_epoch(val_loader, train=False, epoch=epoch)
        is_best = tracker.update(val_metrics["loss"], epoch, {"frequency_cnn": freq_stream.model, "fusion_head": fusion})
        marker = "  (best so far, saved)" if is_best else ""
        gap = val_metrics["clean_acc"] - val_metrics["aug_acc"]
        print(f"epoch {epoch}/{epochs}  train_loss={train_metrics['loss']:.4f}  train_acc={train_metrics['acc']:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}  "
              f"(clean={val_metrics['clean_acc']:.4f}  augmented={val_metrics['aug_acc']:.4f}  gap={gap:+.4f}){marker}")

    # A gated run gets its OWN checkpoint dir (..._gated) rather than
    # overwriting checkpoints/v2_augmented_<mode>/ -- that directory is
    # the already-evaluated, README-documented V2 checkpoint every
    # results/*.json and this project's writeup point at; a retrain that
    # silently clobbered it would make "compare gated vs. non-gated"
    # impossible and could lose the one working checkpoint if the gated
    # run turns out worse. Every downstream script (evaluate.py,
    # calibrate.py, infer.py, explain.py, error_analysis.py) already
    # accepts an explicit --checkpoint path, so pointing them at the
    # gated checkpoint costs nothing extra.
    dir_name = f"v2_augmented_{resolved_mode}" + ("_gated" if use_freq_gate else "")
    ckpt_dir = Path(CHECKPOINT_DIR) / dir_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.pt"
    torch.save(
        {
            "frequency_cnn": tracker.best_state["frequency_cnn"],
            "fusion_head": tracker.best_state["fusion_head"],
            "freq_mode": resolved_mode,
            "best_epoch": tracker.best_epoch,
            "best_val_loss": tracker.best_val_loss,
            "trained_epochs": epochs,
        },
        ckpt_path,
    )
    # Sidecar recording whether this checkpoint's FusionHead was built
    # with use_freq_gate=True -- evaluate.py/calibrate.py/detector.py all
    # read this before constructing a FusionHead to load this state dict
    # into, so a gated and non-gated checkpoint can coexist without
    # either loader needing to be told by hand which is which. See
    # models.fusion's save_architecture_metadata docstring.
    save_architecture_metadata(ckpt_dir, use_freq_gate)
    print(f"\nSaved checkpoint from epoch {tracker.best_epoch}/{epochs} "
          f"(val_loss={tracker.best_val_loss:.4f}) -> {ckpt_path}")
    freq_stream.model.load_state_dict(tracker.best_state["frequency_cnn"])
    fusion.load_state_dict(tracker.best_state["fusion_head"])
    return freq_stream, fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="v0", choices=["v0", "v1", "v2"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4,
                         help="v1/v2 only: DataLoader workers for on-the-fly frequency-input preprocessing")
    parser.add_argument("--freq-mode", default=None, choices=["srm", "fft"],
                         help="v1/v2 only: override config.FREQUENCY_MODE for this run")
    parser.add_argument("--limit", type=int, default=None,
                         help="v2 only: restrict to a random N-image subset (not N rows -- an image's clean "
                              "row and all its augmented rows travel together) for a fast smoke test before "
                              "committing to the full ~4x-larger-than-V1 training set")
    parser.add_argument("--use-freq-gate", action="store_true",
                         help="v2 only: build FusionHead with the residual-energy-conditioned gate on the "
                              "frequency embedding (see models/fusion.py's module docstring) instead of the "
                              "plain V1-style concatenation. Off by default -- every existing checkpoint was "
                              "trained without it, and passing this only affects a fresh training run.")
    args = parser.parse_args()

    if args.stage == "v0":
        train_v0(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    elif args.stage == "v1":
        train_v1(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            freq_mode=args.freq_mode,
        )
    elif args.stage == "v2":
        train_v2(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            freq_mode=args.freq_mode,
            limit=args.limit,
            use_freq_gate=args.use_freq_gate,
        )


if __name__ == "__main__":
    main()
