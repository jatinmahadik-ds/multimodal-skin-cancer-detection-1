"""
STEP 13 - READ-ONLY duplicate verification on the CLEANED ISIC 2019 dataset.

Scans every JPG under dataset_split/{train,validation,test}/{cancer,non_cancer},
computes SHA-256 for each, and reports:
  - within-split duplicates
  - train <-> validation duplicates
  - train <-> test duplicates
  - validation <-> test duplicates

Also cross-checks actual JPG file counts against the CSV manifests.

THIS SCRIPT IS STRICTLY READ-ONLY.
It does not delete, move, rename, or modify anything.
"""

import os
import hashlib
from collections import defaultdict

import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset"
DATASET_DIR = os.path.join(BASE_DIR, "dataset_split")

SPLITS = ["train", "validation", "test"]
CLASSES = ["cancer", "non_cancer"]

LINE = "=" * 70


# ============================================================
# HASH HELPER
# ============================================================

def sha256_of(filepath, chunk=1024 * 1024):
    """Stream the file through SHA-256. Read-only."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ============================================================
# SCAN ALL IMAGES
# ============================================================

print(LINE)
print("STEP 13 - DUPLICATE VERIFICATION (CLEANED DATASET)")
print(LINE)
print("\nMode: READ-ONLY (no files will be modified)")

# hash -> list of (split, class, filename)
hash_index = defaultdict(list)

total_images = 0
per_split_counts = {}

for split in SPLITS:
    split_total = 0
    for class_name in CLASSES:
        folder = os.path.join(DATASET_DIR, split, class_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing folder: {folder}")
        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith(".jpg"):
                continue
            path = os.path.join(folder, filename)
            file_hash = sha256_of(path)
            hash_index[file_hash].append((split, class_name, filename))
            split_total += 1
            total_images += 1
    per_split_counts[split] = split_total
    print(f"  hashed {split}: {split_total}")

# ============================================================
# TOTALS
# ============================================================

duplicate_groups = {h: locs for h, locs in hash_index.items() if len(locs) > 1}

print("\n" + LINE)
print("TOTALS")
print(LINE)
print(f"Total images checked : {total_images}")
print(f"Unique image hashes  : {len(hash_index)}")
print(f"Duplicate groups     : {len(duplicate_groups)}")
print(f"Excess copies        : {total_images - len(hash_index)}")

# ============================================================
# PAIRWISE CLASSIFICATION
# ============================================================

PAIR_KEYS = [
    ("train", "validation", "TRAIN <-> VALIDATION"),
    ("train", "test", "TRAIN <-> TEST"),
    ("validation", "test", "VALIDATION <-> TEST"),
]

pair_counts = {}
for a, b, _label in PAIR_KEYS:
    pair_counts[(a, b)] = 0

within_counts = {s: 0 for s in SPLITS}
within_detail = {s: [] for s in SPLITS}
pair_detail = {label: [] for _a, _b, label in PAIR_KEYS}
cross_label_conflicts = []

for file_hash, locs in duplicate_groups.items():
    splits_present = {loc[0] for loc in locs}

    # within-split duplicates (a split contributing 2+ copies)
    for split in SPLITS:
        n = sum(1 for loc in locs if loc[0] == split)
        if n > 1:
            within_counts[split] += 1
            within_detail[split].append((file_hash, [l for l in locs if l[0] == split]))

    # cross-split pairs
    for a, b, label in PAIR_KEYS:
        if a in splits_present and b in splits_present:
            pair_counts[(a, b)] += 1
            pair_detail[label].append((file_hash, locs))

    # same pixels, contradictory target
    if len({loc[1] for loc in locs}) > 1:
        cross_label_conflicts.append((file_hash, locs))

print("\n" + LINE)
print("DUPLICATES BY CATEGORY")
print(LINE)
print("\nWithin-split duplicate groups:")
for split in SPLITS:
    print(f"  {split:11s}: {within_counts[split]}")

print("\nCross-split duplicate groups:")
print(f"  Train <-> Validation : {pair_counts[('train', 'validation')]}")
print(f"  Train <-> Test       : {pair_counts[('train', 'test')]}")
print(f"  Validation <-> Test  : {pair_counts[('validation', 'test')]}")

print("\nLabel conflicts (identical pixels, different target): "
      f"{len(cross_label_conflicts)}")

# ============================================================
# DETAIL LISTINGS (report only, never removed)
# ============================================================

if duplicate_groups:
    print("\n" + LINE)
    print("DETAIL - ALL DUPLICATE GROUPS (NOT REMOVED)")
    print(LINE)

    idx = 0
    for split in SPLITS:
        for file_hash, locs in sorted(within_detail[split], key=lambda x: str(x[1])):
            idx += 1
            print(f"\n[{idx}] WITHIN {split}  hash={file_hash[:16]}...")
            for s, c, fn in locs:
                print(f"      {s}/{c}/{fn}")

    for _a, _b, label in PAIR_KEYS:
        for file_hash, locs in sorted(pair_detail[label], key=lambda x: str(x[1])):
            idx += 1
            print(f"\n[{idx}] {label}  hash={file_hash[:16]}...")
            for s, c, fn in locs:
                print(f"      {s}/{c}/{fn}")
else:
    print("\n" + LINE)
    print("DETAIL")
    print(LINE)
    print("\nNone. No duplicate groups to list.")

# ============================================================
# CSV MANIFEST vs DISK CROSS-CHECK
# ============================================================

print("\n" + LINE)
print("CSV MANIFEST vs DISK CROSS-CHECK")
print(LINE)

manifest_ok = True
print(f"\n{'split':<12}{'manifest':>10}{'disk':>8}{'match':>8}   cancer / non-cancer (manifest vs disk)")

for split in SPLITS:
    csv_path = os.path.join(DATASET_DIR, f"{split}.csv")
    if not os.path.exists(csv_path):
        print(f"{split:<12}  MISSING MANIFEST ({csv_path})")
        manifest_ok = False
        continue

    manifest = pd.read_csv(csv_path)
    m_cancer = int((manifest["target"] == 1).sum())
    m_non = int((manifest["target"] == 0).sum())

    d_cancer = len([f for f in os.listdir(os.path.join(DATASET_DIR, split, "cancer"))
                    if f.lower().endswith(".jpg")])
    d_non = len([f for f in os.listdir(os.path.join(DATASET_DIR, split, "non_cancer"))
                 if f.lower().endswith(".jpg")])
    disk = d_cancer + d_non

    counts_match = (len(manifest) == disk == per_split_counts[split])
    classes_match = (m_cancer == d_cancer and m_non == d_non)
    ok = counts_match and classes_match
    manifest_ok = manifest_ok and ok

    print(f"{split:<12}{len(manifest):>10}{disk:>8}{'OK' if ok else 'MISMATCH':>8}"
          f"   {m_cancer}/{d_cancer}  {m_non}/{d_non}")

# ============================================================
# VERDICT
# ============================================================

print("\n" + LINE)
print("VERDICT")
print(LINE)

no_dupes = len(duplicate_groups) == 0

if no_dupes and manifest_ok:
    print("\nPASS")
    print("No duplicate images exist across or within the cleaned splits.")
    print("CSV manifests agree with the JPG files on disk.")
elif no_dupes and not manifest_ok:
    print("\nPASS (duplicates) / FAIL (manifest mismatch)")
    print("No duplicate images found, but manifests disagree with disk counts.")
else:
    print("\nFAIL")
    print(f"{len(duplicate_groups)} duplicate group(s) still present. "
          "NOT removed - manual review required.")

print("\nThis script made no changes. Nothing was deleted, moved, renamed or modified.")
print("balanced_train.csv was NOT touched.")
print(LINE)
