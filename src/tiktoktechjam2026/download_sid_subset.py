from datasets import load_dataset
from pathlib import Path

OUT_DIR = Path("data/sid_subset")
N_REAL = 1000
N_FAKE = 1000

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "REAL").mkdir(exist_ok=True)
(OUT_DIR / "FAKE").mkdir(exist_ok=True)

dataset = load_dataset(
    "saberzl/SID_Set",
    split="validation",
    streaming=True,
)

real_count = 0
fake_count = 0

for row in dataset:
    label = row["label"]

    if label == 0 and real_count < N_REAL:
        image = row["image"].convert("RGB")
        image.save(OUT_DIR / "REAL" / f"real_{real_count:04d}.jpg")
        real_count += 1

    elif label == 1 and fake_count < N_FAKE:
        image = row["image"].convert("RGB")
        image.save(OUT_DIR / "FAKE" / f"fake_{fake_count:04d}.jpg")
        fake_count += 1

    if real_count >= N_REAL and fake_count >= N_FAKE:
        break

print(f"Saved {real_count} real images")
print(f"Saved {fake_count} synthetic images")
print(f"Dataset saved to {OUT_DIR}")