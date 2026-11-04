import os
import hashlib
from collections import defaultdict

BASE_DIR = r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset"
DATASET_DIR = os.path.join(BASE_DIR, "dataset_split")
SPLITS = ["train", "validation", "test"]
CLASSES = ["cancer", "non_cancer"]


def get_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


hash_locations = defaultdict(list)

for split in SPLITS:
    for class_name in CLASSES:
        folder = os.path.join(DATASET_DIR, split, class_name)
        for filename in os.listdir(folder):
            if not filename.lower().endswith(".jpg"):
                continue
            filepath = os.path.join(folder, filename)
            hash_locations[get_hash(filepath)].append((split, class_name, filename))

print("=" * 70)
print("CROSS-SPLIT DUPLICATE ANALYSIS")
print("=" * 70)

cross_split = []
label_conflicts = []

for file_hash, locations in hash_locations.items():
    if len(locations) <= 1:
        continue
    splits_found = {loc[0] for loc in locations}
    if len(splits_found) > 1:
        cross_split.append((file_hash, locations))
        classes_found = {loc[1] for loc in locations}
        if len(classes_found) > 1:
            label_conflicts.append((file_hash, locations))

print(f"\nTotal duplicate groups: {sum(1 for v in hash_locations.values() if len(v) > 1)}")
print(f"Cross-split duplicate groups: {len(cross_split)}")
print(f"  ...of which LABEL CONFLICTS (same image, different class): {len(label_conflicts)}")

print("\n" + "=" * 70)
print("ALL CROSS-SPLIT DUPLICATES (leakage into test/val = invalidates scores)")
print("=" * 70)

for i, (h, locs) in enumerate(sorted(cross_split, key=lambda x: str(x[1])), 1):
    conflict = "  <-- LABEL CONFLICT" if len({l[1] for l in locs}) > 1 else ""
    print(f"\nGroup {i}  hash={h[:16]}...{conflict}")
    for split, cls, fn in locs:
        print(f"    {split:12s} {cls:12s} {fn}")

print("\n" + "=" * 70)
print("SUMMARY OF LEAKAGE IMPACT")
print("=" * 70)

train_test_leak = [g for g in cross_split if "test" in {l[0] for l in g[1]}]
train_val_leak = [g for g in cross_split if "validation" in {l[0] for l in g[1]}]

print(f"Images duplicated between TRAIN and TEST:        {len(train_test_leak)}")
print(f"Images duplicated between TRAIN and VALIDATION:  {len(train_val_leak)}")
print(f"Images with conflicting labels (same pixels, different target): {len(label_conflicts)}")

test_leaked_ids = sorted({fn for g in train_test_leak for s, c, fn in g[1] if s == "test"})
print(f"\nTest-set filenames that are exact copies of training images: {len(test_leaked_ids)}")
print("These must be removed or reassigned before final evaluation.")
