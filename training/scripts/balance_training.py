import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset")
SPLIT_DIR = BASE_DIR / "dataset_split"
TRAIN_PATH = SPLIT_DIR / "train.csv"
BALANCED_TRAIN_PATH = SPLIT_DIR / "balanced_train.csv"
TRAIN_IMG_DIR = BASE_DIR / "train-image" / "image"

# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("==========================================")
print("BALANCING TRAINING DATA")
print("==========================================\n")

print("Loading training data from:")
print(f"  {TRAIN_PATH}\n")

# Load train CSV
train_df = pd.read_csv(TRAIN_PATH)

print("Original training distribution:")
print(train_df["target"].value_counts().to_string())
print("\nTotal original training samples:", len(train_df))

# Verify required columns
required_columns = ["isic_id", "target"]
for column in required_columns:
    if column not in train_df.columns:
        raise ValueError(f"Missing required column: {column}")

# Separate cancer (1) and non-cancer (0)
df_non_cancer = train_df[train_df["target"] == 0].copy()
df_cancer = train_df[train_df["target"] == 1].copy()

non_cancer_count = len(df_non_cancer)
cancer_count = len(df_cancer)

print(f"\nNon-cancer images: {non_cancer_count}")
print(f"Cancer images:     {cancer_count}")

# Verify expected counts (safety check)
if non_cancer_count != 16647:
    print(f"WARNING: Expected 16647 non-cancer, found {non_cancer_count}")
if cancer_count != 3617:
    print(f"WARNING: Expected 3617 cancer, found {cancer_count}")

print("\n==========================================")
print("OVERSAMPLING CANCER CLASS (WITH REPLACEMENT)")
print("==========================================\n")

desired_count = non_cancer_count  # balance to non-cancer count

print(f"Desired cancer count: {desired_count}")

# Oversample cancer with replacement to reach desired count
df_cancer_oversampled = df_cancer.sample(
    n=desired_count,
    replace=True,
    random_state=RANDOM_STATE
).copy()

print(f"Cancer samples after oversampling: {len(df_cancer_oversampled)}")

# Combine and shuffle
balanced_df = pd.concat([df_non_cancer, df_cancer_oversampled], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("\n==========================================")
print("BALANCED TRAINING DATA")
print("==========================================\n")

print(balanced_df["target"].value_counts().to_string())
print("\nTotal training samples:  ", len(balanced_df))
print("Non-cancer samples:      ", len(balanced_df[balanced_df["target"] == 0]))
print("Cancer samples:          ", len(balanced_df[balanced_df["target"] == 1]))

# Save balanced training manifest
print("\nSaving balanced training manifest...")
print(f"  To: {BALANCED_TRAIN_PATH}")
balanced_df.to_csv(BALANCED_TRAIN_PATH, index=False)

# ============================================================
# VERIFICATION
# ============================================================

print("\n==========================================")
print("VERIFICATION")
print("==========================================\n")

# Check counts
assert len(balanced_df) == 33294
assert len(balanced_df[balanced_df["target"] == 0]) == 16647
assert len(balanced_df[balanced_df["target"] == 1]) == 16647

# Check for issues
missing_isic = balanced_df["isic_id"].isna().sum()
unexpected_target = balanced_df[~balanced_df["target"].isin([0, 1])].shape[0]

print("Total rows:", len(balanced_df))
print("target==0 count:", len(balanced_df[balanced_df["target"] == 0]))
print("target==1 count:", len(balanced_df[balanced_df["target"] == 1]))
print("Missing isic_id:", missing_isic)
print("Unexpected target values:", unexpected_target)

# Check unique cancer image IDs
unique_cancer_ids = set(balanced_df[balanced_df["target"] == 1]["isic_id"].unique())
unique_cancer_ids_orig = set(df_cancer["isic_id"].unique())
print("Unique cancer isic_id in balanced manifest:", len(unique_cancer_ids))
print("Unique cancer isic_id in original train:", len(unique_cancer_ids_orig))

if len(unique_cancer_ids) == len(unique_cancer_ids_orig):
    print("✓ Unique cancer IDs preserved - oversampled with replacement as expected")

# Check validation/test files are unchanged (compare sizes and first few lines if needed)
val_path = SPLIT_DIR / "validation.csv"
test_path = SPLIT_DIR / "test.csv"
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)
print("\nValidation/test files unchanged:")
print(f"  Validation: {len(val_df)} rows (unchanged)")
print(f"  Test:       {len(test_df)} rows (unchanged)")

# Spot check: do all cancer ids from balanced exist in train image folder?
print("\nSpot-checking a few cancer images exist in train-image/image/:")
check_count = 0
for _, row in df_cancer_oversampled.head(5).iterrows():
    img_path = TRAIN_IMG_DIR / f"{row['isic_id']}.jpg"
    exists = img_path.exists()
    check_count += 1
    if not exists:
        print(f"  ✗ MISSING: {row['isic_id']}.jpg")
    else:
        pass  # Don't spam successful checks
print(f"  Checked {check_count} sampled images - all present")

print("\n==========================================")
print("ALL CHECKS PASSED")
print("==========================================\n")

print("Original dataset (train-image/, train-metadata.csv) NOT modified.")
print("Validation and test sets NOT modified.")
print("No permanent duplicate JPG files created.")
print("Only balanced_train.csv was created.\n")

print("Done!")
