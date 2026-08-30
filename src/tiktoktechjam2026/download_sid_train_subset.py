from datasets import load_dataset
from pathlib import Path


# ==================================================
# Configuration
# ==================================================

OUT_DIR = Path("data/sid_train_subset")

# These were already used in your zero-shot test subset
SKIP_REAL = 1000
SKIP_FAKE = 1000

# Number of NEW images to collect for training
N_REAL = 3000
N_FAKE = 3000


# ==================================================
# Prepare output folders
# ==================================================

REAL_DIR = OUT_DIR / "REAL"
FAKE_DIR = OUT_DIR / "FAKE"

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# Load SID_Set as a stream
# ==================================================

print("Loading SID_Set validation split...")

dataset = load_dataset(
    "saberzl/SID_Set",
    split="validation",
    streaming=True,
)


# ==================================================
# Counters
# ==================================================

seen_real = 0
seen_fake = 0

saved_real = 0
saved_fake = 0


# ==================================================
# Iterate through SID_Set
# ==================================================

print()
print("Creating non-overlapping SID training subset...")
print(f"Skipping first {SKIP_REAL} REAL images")
print(f"Skipping first {SKIP_FAKE} FAKE images")
print(f"Saving next {N_REAL} REAL images")
print(f"Saving next {N_FAKE} FAKE images")
print()


for row in dataset:

    label = row["label"]

    # ==================================================
    # REAL = 0
    # ==================================================

    if label == 0:

        seen_real += 1

        # Skip images already used in zero-shot test
        if seen_real <= SKIP_REAL:
            continue

        if saved_real < N_REAL:

            image = row["image"].convert("RGB")

            save_path = (
                REAL_DIR
                / f"real_{saved_real:05d}.jpg"
            )

            image.save(
                save_path,
                quality=95,
            )

            saved_real += 1

    # ==================================================
    # FULL SYNTHETIC = 1
    # ==================================================

    elif label == 1:

        seen_fake += 1

        # Skip images already used in zero-shot test
        if seen_fake <= SKIP_FAKE:
            continue

        if saved_fake < N_FAKE:

            image = row["image"].convert("RGB")

            save_path = (
                FAKE_DIR
                / f"fake_{saved_fake:05d}.jpg"
            )

            image.save(
                save_path,
                quality=95,
            )

            saved_fake += 1

    # ==================================================
    # Progress
    # ==================================================

    total_saved = (
        saved_real
        + saved_fake
    )

    if (
        total_saved > 0
        and total_saved % 500 == 0
    ):

        print(
            f"Saved {saved_real}/{N_REAL} REAL, "
            f"{saved_fake}/{N_FAKE} FAKE"
        )

    # ==================================================
    # Stop once enough images collected
    # ==================================================

    if (
        saved_real >= N_REAL
        and saved_fake >= N_FAKE
    ):
        break


# ==================================================
# Final summary
# ==================================================

print()
print("=" * 60)
print("SID TRAINING SUBSET COMPLETE")
print("=" * 60)

print(
    f"Saved REAL images: {saved_real}"
)

print(
    f"Saved FAKE images: {saved_fake}"
)

print(
    f"Total images:      "
    f"{saved_real + saved_fake}"
)

print(
    f"Saved to:          {OUT_DIR}"
)