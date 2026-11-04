import os
import hashlib
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


hash_locations = defaultdict(list)

total_images = 0


print("==========================================")
print("ISIC 2019 DUPLICATE CHECK")
print("==========================================")


for split in SPLITS:

    for class_name in CLASSES:

        folder = os.path.join(
            DATASET_DIR,
            split,
            class_name
        )

        print(f"\nChecking: {split}/{class_name}")

        for filename in os.listdir(folder):

            if not filename.lower().endswith(".jpg"):
                continue

            filepath = os.path.join(
                folder,
                filename
            )

            file_hash = get_hash(filepath)

            hash_locations[file_hash].append(
                (split, class_name, filename)
            )

            total_images += 1


# ============================================================
# FIND DUPLICATES
# ============================================================

duplicate_groups = []

for file_hash, locations in hash_locations.items():

    if len(locations) > 1:

        duplicate_groups.append(
            (file_hash, locations)
        )


# ============================================================
# RESULTS
# ============================================================

print("\n==========================================")
print("RESULTS")
print("==========================================")

print("Total images checked:", total_images)

print(
    "Unique image hashes:",
    len(hash_locations)
)

print(
    "Duplicate groups:",
    len(duplicate_groups)
)


# ============================================================
# CHECK CROSS-SPLIT DUPLICATES
# ============================================================

cross_split_groups = []

for file_hash, locations in duplicate_groups:

    splits_found = set(
        location[0]
        for location in locations
    )

    if len(splits_found) > 1:

        cross_split_groups.append(
            (file_hash, locations)
        )


print(
    "Duplicates appearing across splits:",
    len(cross_split_groups)
)


# ============================================================
# DISPLAY EXAMPLES
# ============================================================

if len(duplicate_groups) > 0:

    print("\nDuplicate examples:")

    for i, (file_hash, locations) in enumerate(
        duplicate_groups[:20],
        start=1
    ):

        print(f"\nDuplicate group {i}:")

        for location in locations:

            print(
                "  ",
                location
            )


if len(cross_split_groups) > 0:

    print("\n==========================================")
    print("WARNING")
    print("==========================================")

    print(
        "Some identical images appear in different splits."
    )

    print(
        "DO NOT train the final model yet."
    )

else:

    print("\n==========================================")
    print("PASS")
    print("==========================================")

    print(
        "No exact duplicate images were found across splits."
    )

    print(
        "The dataset can proceed to the next cleaning stage."
    )
