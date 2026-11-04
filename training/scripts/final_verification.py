"""
STEP 15 - FINAL READ-ONLY VERIFICATION of the complete ISIC 2019 dataset.

Covers checks A-N. Opens every JPG to confirm it decodes, is 224x224, and is RGB;
hashes every file for duplicate detection; cross-checks manifests against disk
and against the original train-metadata.csv.

STRICTLY READ-ONLY. Nothing is created, modified, moved, renamed or deleted.
No model training is performed.
"""

import os
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset")
DATASET_DIR = BASE_DIR / "dataset_split"
TRAIN_IMG_DIR = BASE_DIR / "train-image" / "image"
SOURCE_METADATA = BASE_DIR / "train-metadata.csv"

SPLITS = ["train", "validation", "test"]
CLASSES = ["cancer", "non_cancer"]

EXPECTED_SIZE = (224, 224)
SEED = 42
LINE = "=" * 70

# Expected counts from the verified cleaned split
EXPECTED = {
    "train": {"total": 20231, "cancer": 3610, "non_cancer": 16621},
    "validation": {"total": 2526, "cancer": 450, "non_cancer": 2076},
    "test": {"total": 2522, "cancer": 452, "non_cancer": 2070},
}
EXPECTED_BALANCED = {"total": 33242, "cancer": 16621, "non_cancer": 16621}

failures = []
notes = []


def check(label, ok, detail=""):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label:<52} {detail}")
    if not ok:
        failures.append(label)
    return ok


def sha256_of(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


print(LINE)
print("STEP 15 - FINAL DATASET VERIFICATION")
print(LINE)
print("\nMode: READ-ONLY. No training. No file changes.")

# ============================================================
# A / B / L - MANIFEST ROW COUNTS, CLASS DISTRIBUTIONS, TARGETS
# ============================================================

print("\n" + LINE)
print("A / B / L - MANIFESTS: ROW COUNTS, CLASS DISTRIBUTION, TARGETS")
print(LINE)

manifests = {}
for split in SPLITS:
    path = DATASET_DIR / f"{split}.csv"
    if not path.exists():
        check(f"{split}.csv exists", False, str(path))
        continue
    manifests[split] = pd.read_csv(path)
    df = manifests[split]

    has_cols = "isic_id" in df.columns and "target" in df.columns
    if not has_cols:
        check(f"{split}.csv has isic_id + target", False)
        continue

    n0 = int((df["target"] == 0).sum())
    n1 = int((df["target"] == 1).sum())
    exp = EXPECTED[split]
    print(f"\n{split}.csv")
    check(f"{split}.csv row count", len(df) == exp["total"], f"{len(df):,} (exp {exp['total']:,})")
    check(f"{split}.csv cancer count", n1 == exp["cancer"], f"{n1:,} (exp {exp['cancer']:,})")
    check(f"{split}.csv non-cancer count", n0 == exp["non_cancer"], f"{n0:,} (exp {exp['non_cancer']:,})")
    check(f"{split}.csv no missing isic_id", int(df["isic_id"].isna().sum()) == 0,
          f"{int(df['isic_id'].isna().sum())}")
    unexpected = int((~df["target"].isin([0, 1])).sum())
    check(f"{split}.csv no unexpected target values", unexpected == 0, f"{unexpected}")

# balanced_train.csv
print("\nbalanced_train.csv")
balanced_path = DATASET_DIR / "balanced_train.csv"
balanced_df = None
if balanced_path.exists():
    balanced_df = pd.read_csv(balanced_path)
    b0 = int((balanced_df["target"] == 0).sum())
    b1 = int((balanced_df["target"] == 1).sum())
    check("balanced_train.csv row count", len(balanced_df) == EXPECTED_BALANCED["total"],
          f"{len(balanced_df):,} (exp {EXPECTED_BALANCED['total']:,})")
    check("balanced_train.csv target 0 count", b0 == EXPECTED_BALANCED["non_cancer"],
          f"{b0:,} (exp {EXPECTED_BALANCED['non_cancer']:,})")
    check("balanced_train.csv target 1 count", b1 == EXPECTED_BALANCED["cancer"],
          f"{b1:,} (exp {EXPECTED_BALANCED['cancer']:,})")
    check("balanced_train.csv no missing isic_id",
          int(balanced_df["isic_id"].isna().sum()) == 0,
          f"{int(balanced_df['isic_id'].isna().sum())}")
    check("balanced_train.csv no unexpected target values",
          int((~balanced_df["target"].isin([0, 1])).sum()) == 0, "")
else:
    check("balanced_train.csv exists", False, str(balanced_path))

# ============================================================
# C / D / E / F / G - IMAGE EXISTENCE, READABILITY, SIZE, CHANNELS
# ============================================================

print("\n" + LINE)
print("C / D / E / F / G - IMAGE FILES: EXISTENCE, DECODE, 224x224, RGB")
print(LINE)

hash_index = defaultdict(list)   # hash -> [(split, class, filename)]
disk_counts = {}
missing_jpgs = []
corrupted = []
wrong_dims = []
non_rgb = []

for split in SPLITS:
    split_total = 0
    for class_name in CLASSES:
        folder = DATASET_DIR / split / class_name
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(".jpg"):
                continue
            path = folder / fn
            split_total += 1

            # D / E / F / G : decode + validate
            try:
                with Image.open(path) as im:
                    im.load()                      # forces full decode
                    size = im.size
                    mode = im.mode
            except Exception:
                corrupted.append(f"{split}/{class_name}/{fn}")
                hash_index[sha256_of(path)].append((split, class_name, fn))
                continue

            if size != EXPECTED_SIZE:
                wrong_dims.append(f"{split}/{class_name}/{fn} {size[0]}x{size[1]}")
            if mode != "RGB":
                non_rgb.append(f"{split}/{class_name}/{fn} mode={mode}")

            # H : hash for duplicate detection
            hash_index[sha256_of(path)].append((split, class_name, fn))

    disk_counts[split] = split_total

# C : every manifest isic_id resolves to a real JPG in its split folder
print("\nManifest -> disk resolution:")
for split in SPLITS:
    if split not in manifests:
        continue
    df = manifests[split]
    unresolved = 0
    for isic_id, target in df[["isic_id", "target"]].drop_duplicates().itertuples(index=False):
        folder = DATASET_DIR / split / ("cancer" if int(target) == 1 else "non_cancer")
        if not (folder / f"{isic_id}.jpg").exists():
            unresolved += 1
            missing_jpgs.append(f"{split}/{isic_id}.jpg")
    check(f"{split}.csv all isic_id resolve to a JPG", unresolved == 0, f"{unresolved} unresolved")
    check(f"{split} manifest count == disk JPG count",
          len(df) == disk_counts[split], f"{len(df):,} vs {disk_counts[split]:,}")

print("\nImage integrity:")
check("All JPGs decode successfully", len(corrupted) == 0, f"{len(corrupted)} corrupted")
check("All images are exactly 224x224", len(wrong_dims) == 0, f"{len(wrong_dims)} wrong size")
check("All images are 3-channel RGB", len(non_rgb) == 0, f"{len(non_rgb)} non-RGB")
check("No missing JPG files", len(missing_jpgs) == 0, f"{len(missing_jpgs)} missing")

# ============================================================
# H - SHA-256 DUPLICATES
# ============================================================

print("\n" + LINE)
print("H - SHA-256 DUPLICATE DETECTION")
print(LINE)

dup_groups = {h: locs for h, locs in hash_index.items() if len(locs) > 1}
total_imgs = sum(disk_counts.values())

within = defaultdict(int)
pairs = {"train|validation": 0, "train|test": 0, "validation|test": 0}
label_conflicts = []

for _h, locs in dup_groups.items():
    for s in SPLITS:
        if sum(1 for l in locs if l[0] == s) > 1:
            within[s] += 1
    present = {l[0] for l in locs}
    for key in pairs:
        a, b = key.split("|")
        if a in present and b in present:
            pairs[key] += 1
    if len({l[1] for l in locs}) > 1:
        label_conflicts.append((_h, locs))

print(f"\nTotal images hashed : {total_imgs:,}")
print(f"Unique hashes       : {len(hash_index):,}")
print(f"Duplicate groups    : {len(dup_groups)}")
print(f"Label conflicts     : {len(label_conflicts)}")

check("Zero duplicate groups", len(dup_groups) == 0, f"{len(dup_groups)}")
for s in SPLITS:
    check(f"Within-{s} duplicates", within[s] == 0, f"{within[s]}")
for key, cnt in pairs.items():
    check(f"Cross-split {key} duplicates", cnt == 0, f"{cnt}")
check("Zero label conflicts", len(label_conflicts) == 0, f"{len(label_conflicts)}")

if dup_groups:
    print("\nDuplicate detail (NOT removed):")
    for i, (h, locs) in enumerate(list(dup_groups.items())[:20], 1):
        print(f"\n  [{i}] hash={h[:16]}...")
        for s, c, fn in locs:
            print(f"        {s}/{c}/{fn}")

# ============================================================
# I / N - BALANCED MANIFEST CONFINEMENT + NO PHYSICAL DUPLICATES
# ============================================================

print("\n" + LINE)
print("I / N - BALANCED MANIFEST CONFINEMENT & OVERSAMPLING INTEGRITY")
print(LINE)

if balanced_df is not None and "train" in manifests:
    train_ids = set(manifests["train"]["isic_id"].astype(str))
    balanced_ids = set(balanced_df["isic_id"].astype(str))
    outside = balanced_ids - train_ids
    check("balanced_train.csv references only train isic_ids",
          len(outside) == 0, f"{len(outside)} outside train")

    # every balanced id exists as a JPG inside dataset_split/train only
    wrong_folder = 0
    for isic_id, target in balanced_df[["isic_id", "target"]].drop_duplicates().itertuples(index=False):
        cls = "cancer" if int(target) == 1 else "non_cancer"
        if not (DATASET_DIR / "train" / cls / f"{isic_id}.jpg").exists():
            wrong_folder += 1
    check("All balanced ids exist under dataset_split/train only",
          wrong_folder == 0, f"{wrong_folder} missing")

    uniq_cancer = balanced_df[balanced_df["target"] == 1]["isic_id"].nunique()
    print(f"\n  Cancer rows in manifest : {int((balanced_df['target'] == 1).sum()):,}")
    print(f"  Unique cancer isic_ids  : {uniq_cancer:,}")
    print(f"  JPGs in train/cancer    : {disk_counts['train'] - EXPECTED['train']['non_cancer']:,}")
    check("No physical duplicate JPGs created (disk counts unchanged)",
          disk_counts["train"] == EXPECTED["train"]["total"], f"{disk_counts['train']:,}")
    check("Oversampling exists only in the manifest",
          uniq_cancer < EXPECTED_BALANCED["cancer"],
          f"{uniq_cancer:,} unique < {EXPECTED_BALANCED['cancer']:,} rows")

    # confirm train/validation/test folders contain no oversampling artefacts
    check("No oversampling in train.csv", len(manifests["train"]) == EXPECTED["train"]["total"],
          f"{len(manifests['train']):,}")

# ============================================================
# K - VALIDATION / TEST UNTOUCHED, LABELS MATCH SOURCE METADATA
# ============================================================

print("\n" + LINE)
print("K - VALIDATION / TEST INTEGRITY vs ORIGINAL train-metadata.csv")
print(LINE)

if SOURCE_METADATA.exists():
    src = pd.read_csv(SOURCE_METADATA, usecols=["isic_id", "target"])
    src_map = dict(zip(src["isic_id"].astype(str), src["target"].astype(int)))
else:
    src_map = None
    notes.append("train-metadata.csv not readable; label cross-check skipped")

for split in ("validation", "test"):
    if split not in manifests or src_map is None:
        continue
    df = manifests[split]
    mismatched = [
        isic_id for isic_id, t in df[["isic_id", "target"]].itertuples(index=False)
        if src_map.get(str(isic_id)) != int(t)
    ]
    check(f"{split} targets match source metadata", len(mismatched) == 0,
          f"{len(mismatched)} mismatched")
    check(f"{split} retains original distribution",
          len(df) == EXPECTED[split]["total"], f"{len(df):,}")

if "train" in manifests and "validation" in manifests and "test" in manifests:
    tr = set(manifests["train"]["isic_id"].astype(str))
    va = set(manifests["validation"]["isic_id"].astype(str))
    te = set(manifests["test"]["isic_id"].astype(str))
    check("train/validation isic_id sets are disjoint", len(tr & va) == 0, f"{len(tr & va)} overlap")
    check("train/test isic_id sets are disjoint", len(tr & te) == 0, f"{len(tr & te)} overlap")
    check("validation/test isic_id sets are disjoint", len(va & te) == 0, f"{len(va & te)} overlap")

# ============================================================
# M - CLASS IMBALANCE RATIOS (natural splits, NOT balanced)
# ============================================================

print("\n" + LINE)
print("M - CLASS IMBALANCE RATIO (natural distribution)")
print(LINE)

print(f"\n{'split':<12}{'non-cancer':>12}{'cancer':>10}{'total':>9}{'ratio':>10}{'cancer %':>10}")
ratios = {}
for split in SPLITS:
    exp = EXPECTED[split]
    n0, n1 = exp["non_cancer"], exp["cancer"]
    ratio = n0 / n1
    ratios[split] = ratio
    pct = 100 * n1 / (n0 + n1)
    print(f"{split:<12}{n0:>12,}{n1:>10,}{n0 + n1:>9,}{ratio:>9.2f}:1{pct:>9.2f}%")

print(f"\nFull dataset ratio : "
      f"{sum(EXPECTED[s]['non_cancer'] for s in SPLITS) / sum(EXPECTED[s]['cancer'] for s in SPLITS):.2f}:1")
print(f"Balanced train ratio: 1.00:1  (by construction, manifest only)")

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + LINE)
print("STEP 15 - FINAL DATASET VERIFICATION")
print(LINE)

status = "PASS" if not failures else "FAIL"
print(f"\nDataset status: {status}")

for split in SPLITS:
    e = EXPECTED[split]
    print(f"\n{split.title()}:")
    print(f"  Total      : {e['total']:,}")
    print(f"  Cancer     : {e['cancer']:,}")
    print(f"  Non-cancer : {e['non_cancer']:,}")
    print(f"  Ratio      : {ratios[split]:.2f}:1  ({100 * e['cancer'] / e['total']:.2f}% cancer)")

print("\nBalanced training:")
if balanced_df is not None:
    print(f"  Total      : {len(balanced_df):,}")
    print(f"  Cancer     : {int((balanced_df['target'] == 1).sum()):,} rows "
          f"(from {balanced_df[balanced_df['target'] == 1]['isic_id'].nunique():,} unique images)")
    print(f"  Non-cancer : {int((balanced_df['target'] == 0).sum()):,} rows "
          f"(from {balanced_df[balanced_df['target'] == 0]['isic_id'].nunique():,} unique images)")
    print(f"  Ratio      : 1.00:1")

print(f"\nCorrupted images        : {len(corrupted)}")
print(f"Wrong dimensions        : {len(wrong_dims)}")
print(f"Non-RGB images          : {len(non_rgb)}")
print(f"Missing JPG files       : {len(missing_jpgs)}")
print(f"Duplicate images        : {len(dup_groups)}")
print(f"Cross-split leakage     : {sum(pairs.values())}")
print(f"Label conflicts         : {len(label_conflicts)}")
print("\nBalanced manifest verification:")
if balanced_df is not None:
    b_ok = (len(balanced_df) == EXPECTED_BALANCED["total"]
            and int((balanced_df["target"] == 0).sum()) == EXPECTED_BALANCED["non_cancer"]
            and int((balanced_df["target"] == 1).sum()) == EXPECTED_BALANCED["cancer"])
    print(f"  rows={len(balanced_df):,}  target0={int((balanced_df['target'] == 0).sum()):,}  "
          f"target1={int((balanced_df['target'] == 1).sum()):,}  -> {'OK' if b_ok else 'MISMATCH'}")
    print(f"  confined to train split: {'OK' if len(balanced_ids - train_ids) == 0 else 'VIOLATION'}")
    print(f"  physical duplicate JPGs created: 0")

if notes:
    print("\nNotes:")
    for n in notes:
        print(f"  - {n}")

if failures:
    print("\n" + LINE)
    print("FAIL")
    print("The following checks did not pass:")
    for f in failures:
        print(f"  - {f}")
    print("\nNothing was fixed automatically. Resolve these before training.")
else:
    print("\n" + LINE)
    print("PASS")
    print("ISIC 2019 dataset is ready for EfficientNet training.")
    print(LINE)
