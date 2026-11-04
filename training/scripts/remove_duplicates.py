import os
import shutil
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset_split"
)

REPORT_FILE = os.path.join(
    DATASET_DIR,
    "duplicate_groups_report.csv"
)

# Safety backup location
BACKUP_DIR = os.path.join(
    DATASET_DIR,
    "removed_duplicates_backup"
)

os.makedirs(BACKUP_DIR, exist_ok=True)


# ============================================================
# LOAD DUPLICATE REPORT
# ============================================================

df = pd.read_csv(REPORT_FILE)

print("==========================================")
print("ISIC 2019 DUPLICATE CLEANING")
print("==========================================")


# ============================================================
# GROUP DUPLICATES
# ============================================================

groups = df.groupby(
    ["duplicate_group", "duplicate_type"]
)


removed = []
kept = []


# ============================================================
# PROCESS EACH DUPLICATE GROUP
# ============================================================

for (group_id, duplicate_type), group in groups:

    records = group.to_dict("records")


    # --------------------------------------------------------
    # CONTRADICTORY LABEL
    # --------------------------------------------------------

    if duplicate_type == "CONTRADICTORY_LABEL":

        print(
            f"\nCONTRADICTORY GROUP {group_id}"
        )

        for record in records:

            split = record["split"]
            class_name = record["class"]
            image_id = record["isic_id"]

            print(
                f"  Removing: {split}/{class_name}/{image_id}.jpg"
            )

            source = os.path.join(
                DATASET_DIR,
                split,
                class_name,
                image_id + ".jpg"
            )

            if os.path.exists(source):

                backup_folder = os.path.join(
                    BACKUP_DIR,
                    "contradictory",
                    split,
                    class_name
                )

                os.makedirs(
                    backup_folder,
                    exist_ok=True
                )

                backup_path = os.path.join(
                    backup_folder,
                    image_id + ".jpg"
                )

                shutil.move(
                    source,
                    backup_path
                )

            removed.append({
                "duplicate_group": group_id,
                "duplicate_type": duplicate_type,
                "split": split,
                "class": class_name,
                "isic_id": image_id,
                "reason": "contradictory_label"
            })

        continue


    # --------------------------------------------------------
    # NORMAL DUPLICATES
    # --------------------------------------------------------

    # Prefer TRAIN
    train_records = [
        r for r in records
        if r["split"] == "train"
    ]

    if len(train_records) > 0:

        keep_record = train_records[0]

    else:

        # Otherwise prefer validation
        validation_records = [
            r for r in records
            if r["split"] == "validation"
        ]

        if len(validation_records) > 0:

            keep_record = validation_records[0]

        else:

            keep_record = records[0]


    keep_id = keep_record["isic_id"]

    kept.append({
        "duplicate_group": group_id,
        "duplicate_type": duplicate_type,
        "split": keep_record["split"],
        "class": keep_record["class"],
        "isic_id": keep_id
    })


    print(
        f"\nGroup {group_id} ({duplicate_type})"
    )

    print(
        f"  KEEP: {keep_record['split']}/"
        f"{keep_record['class']}/"
        f"{keep_id}.jpg"
    )


    # --------------------------------------------------------
    # REMOVE ALL OTHER COPIES
    # --------------------------------------------------------

    for record in records:

        if (
            record["split"] == keep_record["split"]
            and record["isic_id"] == keep_record["isic_id"]
        ):
            continue

        split = record["split"]
        class_name = record["class"]
        image_id = record["isic_id"]


        source = os.path.join(
            DATASET_DIR,
            split,
            class_name,
            image_id + ".jpg"
        )


        if os.path.exists(source):

            backup_folder = os.path.join(
                BACKUP_DIR,
                duplicate_type,
                split,
                class_name
            )

            os.makedirs(
                backup_folder,
                exist_ok=True
            )

            backup_path = os.path.join(
                backup_folder,
                image_id + ".jpg"
            )


            shutil.move(
                source,
                backup_path
            )


            print(
                f"  REMOVE: {split}/"
                f"{class_name}/{image_id}.jpg"
            )


            removed.append({
                "duplicate_group": group_id,
                "duplicate_type": duplicate_type,
                "split": split,
                "class": class_name,
                "isic_id": image_id,
                "reason": "duplicate"
            })


# ============================================================
# UPDATE CSV MANIFESTS
# ============================================================

print("\n==========================================")
print("UPDATING CSV FILES")
print("==========================================")


for split in [
    "train",
    "validation",
    "test"
]:

    csv_path = os.path.join(
        DATASET_DIR,
        split + ".csv"
    )

    manifest = pd.read_csv(csv_path)


    # IDs that should be removed from this split
    remove_ids = {
        item["isic_id"]
        for item in removed
        if item["split"] == split
    }


    before = len(manifest)


    manifest = manifest[
        ~manifest["isic_id"].astype(str).isin(remove_ids)
    ].copy()


    after = len(manifest)


    manifest.to_csv(
        csv_path,
        index=False
    )


    print(
        f"{split}: {before} -> {after}"
    )


# ============================================================
# SAVE CLEANING LOG
# ============================================================

removed_df = pd.DataFrame(removed)

removed_df.to_csv(
    os.path.join(
        DATASET_DIR,
        "duplicate_cleaning_log.csv"
    ),
    index=False
)


kept_df = pd.DataFrame(kept)

kept_df.to_csv(
    os.path.join(
        DATASET_DIR,
        "duplicate_kept_log.csv"
    ),
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print("CLEANING COMPLETED")
print("==========================================")

print(
    "Images removed from dataset_split:",
    len(removed)
)

print(
    "Backup location:",
    BACKUP_DIR
)

print(
    "\nIMPORTANT:"
)

print(
    "Original train-image and organized_dataset "
    "were NOT modified."
)

print(
    "\nThe old balanced_train.csv is now OUTDATED."
)

print(
    "We will regenerate it after verifying the cleaned split."
)
