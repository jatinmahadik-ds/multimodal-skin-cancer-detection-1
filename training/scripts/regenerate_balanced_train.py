"""
STEP 14 - Regenerate the balanced training manifest from the CLEANED split.

Reads dataset_split/train.csv (post-duplicate-cleaning), oversamples the cancer
class WITH REPLACEMENT up to the non-cancer count, shuffles with random_state=42,
and overwrites dataset_split/balanced_train.csv.

Oversampling happens ONLY in the CSV. No image files are created, copied,
moved, renamed or deleted. Validation and test are never touched.
"""

import os
from pathlib import Path

import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset")
DATASET_DIR = BASE_DIR / "dataset_split"

TRAIN_CSV = DATASET_DIR / "train.csv"
BALANCED_CSV = DATASET_DIR / "balanced_train.csv"

# Source of truth for verifying images: the cleaned train folders
TRAIN_CANCER_DIR = DATASET_DIR / "train" / "cancer"
TRAIN_NON_CANCER_DIR = DATASET_DIR / "train" / "non_cancer"

SEED = 42
LINE = "=" * 70

# Expected targets (from the cleaned, verified split)
EXPECTED_NON_CANCER = 16621
EXPECTED_CANCER = 3610
EXPECTED_BALANCED_TOTAL = 33242


# ============================================================
# READ CLEAN TRAIN MANIFEST
# ============================================================

print(LINE)
print("STEP 14 - BALANCED TRAINING MANIFEST")
print(LINE)

if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"train.csv not found: {TRAIN_CSV}")

train_df = pd.read_csv(TRAIN_CSV)

for col in ("isic_id", "target"):
    if col not in train_df.columns:
        raise ValueError(f"Missing required column '{col}' in train.csv")

df_non_cancer = train_df[train_df["target"] == 0].copy()
df_cancer = train_df[train_df["target"] == 1].copy()

non_cancer_count = len(df_non_cancer)
cancer_count = len(df_cancer)

print("\nOriginal:")
print(f"  Non-cancer: {non_cancer_count:,}")
print(f"  Cancer:     {cancer_count:,}")
print(f"  Total:      {len(train_df):,}")

# Sanity check against the verified cleaned split
if non_cancer_count != EXPECTED_NON_CANCER or cancer_count != EXPECTED_CANCER:
    print("\n  WARNING: counts differ from the verified cleaned split "
          f"(expected {EXPECTED_NON_CANCER:,} / {EXPECTED_CANCER:,}).")
    print("  Continuing, but verify this is the intended train.csv.")

# ============================================================
# OVERSAMPLE CANCER (WITH REPLACEMENT)
# ============================================================

print("\nOversampling cancer class WITH replacement "
      f"to {non_cancer_count:,} samples...")

df_cancer_balanced = df_cancer.sample(
    n=non_cancer_count,
    replace=True,
    random_state=SEED,
).copy()

print(f"  Cancer samples after oversampling: {len(df_cancer_balanced):,}")

# ============================================================
# COMBINE + SHUFFLE
# ============================================================

balanced_df = (
    pd.concat([df_non_cancer, df_cancer_balanced], ignore_index=True)
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

print("\nBalanced:")
print(f"  Non-cancer: {len(balanced_df[balanced_df['target'] == 0]):,}")
print(f"  Cancer:     {len(balanced_df[balanced_df['target'] == 1]):,}")
print(f"  Total:      {len(balanced_df):,}")

# ============================================================
# SAVE
# ============================================================

balanced_df.to_csv(BALANCED_CSV, index=False)
print(f"\nSaved to:\n  {BALANCED_CSV}")

# ============================================================
# VERIFICATION
# ============================================================

print("\n" + LINE)
print("VERIFICATION")
print(LINE)

checks = []

total_rows = len(balanced_df)
n0 = int((balanced_df["target"] == 0).sum())
n1 = int((balanced_df["target"] == 1).sum())
missing_ids = int(balanced_df["isic_id"].isna().sum())
unexpected = int((~balanced_df["target"].isin([0, 1])).sum())

checks.append(("Total rows == 33,242", total_rows == EXPECTED_BALANCED_TOTAL, f"{total_rows:,}"))
checks.append(("target 0 == 16,621", n0 == EXPECTED_NON_CANCER, f"{n0:,}"))
checks.append(("target 1 == 16,621", n1 == EXPECTED_NON_CANCER, f"{n1:,}"))
checks.append(("No missing isic_id", missing_ids == 0, f"{missing_ids}"))
checks.append(("No unexpected target values", unexpected == 0, f"{unexpected}"))

# Every referenced isic_id must exist as a real JPG in dataset_split/train/<class>/
referenced = balanced_df[["isic_id", "target"]].drop_duplicates()
missing_jpg = []
for isic_id, target in referenced.itertuples(index=False):
    folder = TRAIN_CANCER_DIR if int(target) == 1 else TRAIN_NON_CANCER_DIR
    if not (folder / f"{isic_id}.jpg").exists():
        missing_jpg.append(f"{isic_id}.jpg")

checks.append(("All referenced JPGs exist in dataset_split/train", len(missing_jpg) == 0, f"{len(missing_jpg)}"))

print()
for label, ok, detail in checks:
    print(f"  [{'OK' if ok else 'FAIL'}] {label:45s} {detail}")

# Confirm no duplicate image files were physically created
print("\nOversampling integrity:")
disk_cancer = len([f for f in os.listdir(TRAIN_CANCER_DIR) if f.lower().endswith(".jpg")])
disk_non_cancer = len([f for f in os.listdir(TRAIN_NON_CANCER_DIR) if f.lower().endswith(".jpg")])
uniq_cancer_ids = balanced_df[balanced_df["target"] == 1]["isic_id"].nunique()

print(f"  Unique cancer isic_ids in manifest : {uniq_cancer_ids:,}")
print(f"  JPGs on disk  train/cancer         : {disk_cancer:,}")
print(f"  JPGs on disk  train/non_cancer     : {disk_non_cancer:,}")
print(f"  Manifest rows for cancer           : {n1:,}")
print(f"  -> cancer appears {n1:,} times in the CSV but only {disk_cancer:,} "
      f"JPG files exist on disk.")

no_files_created = (disk_cancer == EXPECTED_CANCER and disk_non_cancer == EXPECTED_NON_CANCER)
checks.append(("No permanent duplicate JPGs created", no_files_created,
               f"{disk_cancer:,} / {disk_non_cancer:,}"))

all_ok = all(ok for _l, ok, _d in checks)

print("\n" + LINE)
print(f"Missing isic_id: {missing_ids}")
print(f"Missing JPG files: {len(missing_jpg)}")
print(f"Unexpected target values: {unexpected}")
print(LINE)

if missing_jpg:
    print("\nMissing JPGs (first 20):")
    for name in missing_jpg[:20]:
        print(f"  {name}")
    if len(missing_jpg) > 20:
        print(f"  ... and {len(missing_jpg) - 20} more")

print()
if all_ok:
    print("PASS")
    print("Balanced training manifest is ready.")
else:
    print("FAIL")
    print("Manifest verification failed - do not train yet.")

print("\nNo images were created, modified, moved or deleted.")
print("train-image/, train-metadata.csv, organized_dataset/ untouched.")
print("validation.csv, test.csv and their folders untouched.")
print("Model training was NOT started.")
print(LINE)
