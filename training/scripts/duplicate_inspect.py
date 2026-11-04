import os
import hashlib
import pandas as pd
from collections import defaultdict

BASE_DIR = r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset_split"
)

SPLITS = ["train", "validation", "test"]
CLASSES = ["cancer", "non_cancer"]


def get_hash(filepath):
    """Calculate SHA-256 hash of an image."""
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as f:
        while True:
            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ------------------------------------------------------------
# Read manifests
# ------------------------------------------------------------

manifests = {}

for split in SPLITS:

    csv_path = os.path.join(
        DATASET_DIR,
        split + ".csv"
    )

    manifests[split] = pd.read_csv(csv_path)


# ------------------------------------------------------------
# Find duplicate hashes
# ------------------------------------------------------------

hash_locations = defaultdict(list)

for split in SPLITS:

    for _, row in manifests[split].iterrows():

        image_id = str(row["isic_id"])
        target = int(row["target"])

        if target == 1:
            class_name = "cancer"
        else:
            class_name = "non_cancer"

        image_path = os.path.join(
            DATASET_DIR,
            split,
            class_name,
            image_id + ".jpg"
        )

        if not os.path.exists(image_path):
            continue

        image_hash = get_hash(image_path)

        hash_locations[image_hash].append({
            "split": split,
            "class": class_name,
            "target": target,
            "isic_id": image_id
        })


# ------------------------------------------------------------
# Keep duplicate groups
# ------------------------------------------------------------

duplicate_groups = []

for image_hash, locations in hash_locations.items():

    if len(locations) > 1:

        duplicate_groups.append({
            "hash": image_hash,
            "locations": locations
        })


# ------------------------------------------------------------
# Create detailed report
# ------------------------------------------------------------

rows = []

for group_number, group in enumerate(
    duplicate_groups,
    start=1
):

    locations = group["locations"]

    targets = set(
        x["target"]
        for x in locations
    )

    splits = set(
        x["split"]
        for x in locations
    )

    if len(targets) > 1:
        duplicate_type = "CONTRADICTORY_LABEL"

    elif len(splits) > 1:
        duplicate_type = "CROSS_SPLIT"

    else:
        duplicate_type = "WITHIN_SPLIT"

    for item in locations:

        rows.append({
            "duplicate_group": group_number,
            "duplicate_type": duplicate_type,
            "hash": group["hash"],
            "split": item["split"],
            "class": item["class"],
            "target": item["target"],
            "isic_id": item["isic_id"]
        })


report = pd.DataFrame(rows)

output_path = os.path.join(
    DATASET_DIR,
    "duplicate_groups_report.csv"
)

report.to_csv(
    output_path,
    index=False
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n==========================================")
print("DUPLICATE INSPECTION")
print("==========================================")

print("Duplicate groups:", len(duplicate_groups))

print("\nDuplicate type:")

print(
    report.groupby(
        ["duplicate_group", "duplicate_type"]
    ).size()
    .reset_index(name="images")
    ["duplicate_type"]
    .value_counts()
)

print("\nReport saved to:")
print(output_path)


# ------------------------------------------------------------
# Contradictory labels
# ------------------------------------------------------------

contradictory = report[
    report["duplicate_type"] == "CONTRADICTORY_LABEL"
]

print("\n==========================================")
print("CONTRADICTORY LABEL GROUPS")
print("==========================================")

if len(contradictory) == 0:

    print("None found.")

else:

    print(
        contradictory[
            [
                "duplicate_group",
                "split",
                "class",
                "target",
                "isic_id"
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# Cross split
# ------------------------------------------------------------

cross_split = report[
    report["duplicate_type"] == "CROSS_SPLIT"
]

print("\n==========================================")
print("CROSS-SPLIT DUPLICATES")
print("==========================================")

if len(cross_split) == 0:

    print("None found.")

else:

    print(
        cross_split[
            [
                "duplicate_group",
                "split",
                "class",
                "target",
                "isic_id"
            ]
        ].to_string(index=False)
    )
