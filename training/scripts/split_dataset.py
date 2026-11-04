import os
import shutil
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset"

METADATA_FILE = os.path.join(BASE_DIR, "train-metadata.csv")

# NOTE: images live in the nested "image" subfolder, i.e. train-image/image/*.jpg
IMAGE_DIR = os.path.join(BASE_DIR, "train-image", "image")

OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_split")


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# 80% train, 10% validation, 10% test


# ============================================================
# READ METADATA
# ============================================================

df = pd.read_csv(METADATA_FILE)

print("==========================================")
print("ISIC 2019 DATASET SPLITTING")
print("==========================================")

print("\nTotal images:", len(df))

print("\nOriginal class distribution:")
print(df["target"].value_counts())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["isic_id", "target"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")


# ============================================================
# VERIFY IMAGE DIRECTORY IS CORRECT (fail fast)
# ============================================================

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

_images_on_disk = len([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".jpg")])
print("\nImages found on disk:", _images_on_disk)

if _images_on_disk != len(df):
    raise ValueError(
        f"Disk/metadata mismatch: {_images_on_disk} images on disk "
        f"vs {len(df)} metadata rows. Fix this before splitting."
    )


# ============================================================
# FIRST SPLIT
# 80% TRAIN
# 20% TEMP
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["target"],
    random_state=RANDOM_STATE
)


# ============================================================
# SECOND SPLIT
# TEMP -> 50% VALIDATION + 50% TEST
#
# Therefore:
# TRAIN = 80%
# VALIDATION = 10%
# TEST = 10%
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["target"],
    random_state=RANDOM_STATE
)


# ============================================================
# PRINT SPLIT INFORMATION
# ============================================================

print("\n==========================================")
print("SPLIT RESULTS")
print("==========================================")

print("\nTRAIN:")
print("Total:", len(train_df))
print(train_df["target"].value_counts())

print("\nVALIDATION:")
print("Total:", len(val_df))
print(val_df["target"].value_counts())

print("\nTEST:")
print("Total:", len(test_df))
print(test_df["target"].value_counts())


# ============================================================
# CREATE FOLDERS
# ============================================================

folders = [
    os.path.join(OUTPUT_DIR, "train", "cancer"),
    os.path.join(OUTPUT_DIR, "train", "non_cancer"),

    os.path.join(OUTPUT_DIR, "validation", "cancer"),
    os.path.join(OUTPUT_DIR, "validation", "non_cancer"),

    os.path.join(OUTPUT_DIR, "test", "cancer"),
    os.path.join(OUTPUT_DIR, "test", "non_cancer"),
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# FUNCTION TO COPY IMAGES
# ============================================================

def copy_images(dataframe, split_name):

    print(f"\nCopying {split_name} images...")

    copied = 0
    missing = 0
    missing_names = []

    for _, row in dataframe.iterrows():

        image_id = str(row["isic_id"])
        target = int(row["target"])

        image_name = image_id + ".jpg"

        source = os.path.join(
            IMAGE_DIR,
            image_name
        )

        # Target folder
        if target == 1:
            class_name = "cancer"
        else:
            class_name = "non_cancer"

        destination_folder = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        destination = os.path.join(
            destination_folder,
            image_name
        )

        # Check image exists
        if not os.path.exists(source):
            missing_names.append(image_name)
            missing += 1
            continue

        # Copy image
        shutil.copy2(source, destination)

        copied += 1

    print(f"{split_name} copied: {copied}")
    print(f"{split_name} missing: {missing}")

    # Only print filenames if something is actually missing (avoid flooding console)
    if missing:
        for name in missing_names[:20]:
            print("  Missing:", name)
        if missing > 20:
            print(f"  ... and {missing - 20} more missing")

    return copied, missing


# ============================================================
# COPY DATA
# ============================================================

train_copied, train_missing = copy_images(
    train_df,
    "train"
)

val_copied, val_missing = copy_images(
    val_df,
    "validation"
)

test_copied, test_missing = copy_images(
    test_df,
    "test"
)


# ============================================================
# SAVE CSV FILES
# ============================================================

train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)


# ============================================================
# FINAL VERIFICATION (count actual files on disk)
# ============================================================

print("\n==========================================")
print("FINAL FOLDER VERIFICATION")
print("==========================================")

verification_ok = True

for split_name in ["train", "validation", "test"]:
    for class_name in ["cancer", "non_cancer"]:
        folder = os.path.join(OUTPUT_DIR, split_name, class_name)
        actual = len([
            f for f in os.listdir(folder)
            if f.lower().endswith(".jpg")
        ])
        print(f"{split_name}/{class_name}: {actual}")

        subset = {"train": train_df, "validation": val_df, "test": test_df}[split_name]
        expected = int((subset["target"] == (1 if class_name == "cancer" else 0)).sum())

        if actual != expected:
            print(f"  WARNING: expected {expected}, found {actual}")
            verification_ok = False

total_missing = train_missing + val_missing + test_missing
if total_missing > 0:
    verification_ok = False

if verification_ok:
    print("\nAll split counts verified successfully.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("SPLITTING COMPLETED")
print("==========================================")

print("Train images:", train_copied)
print("Validation images:", val_copied)
print("Test images:", test_copied)

print(
    "\nTotal copied:",
    train_copied + val_copied + test_copied
)

print("\nTotal missing:", total_missing)

print("\nDataset location:")
print(OUTPUT_DIR)

print("\nOriginal dataset was NOT modified.")
