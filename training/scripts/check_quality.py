import os
import cv2
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset"

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset_split",
    "train"
)

CSV_FILE = os.path.join(
    BASE_DIR,
    "dataset_split",
    "train.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset_split",
    "image_quality_report.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_WIDTH = 224
EXPECTED_HEIGHT = 224

# These are FLAGGING thresholds, NOT automatic deletion rules.
DARK_THRESHOLD = 25
BRIGHT_THRESHOLD = 230
BLUR_THRESHOLD = 30


# ============================================================
# LOAD TRAIN CSV
# ============================================================

df = pd.read_csv(CSV_FILE)

print("==========================================")
print("ISIC 2019 IMAGE QUALITY CHECK")
print("==========================================")

print("Images to check:", len(df))


# ============================================================
# RESULTS
# ============================================================

results = []

corrupted = 0
wrong_size = 0
very_dark = 0
very_bright = 0
very_blurry = 0


# ============================================================
# CHECK EACH IMAGE
# ============================================================

for index, row in df.iterrows():

    image_id = str(row["isic_id"])
    target = int(row["target"])

    filename = image_id + ".jpg"

    if target == 1:
        class_name = "cancer"
    else:
        class_name = "non_cancer"

    image_path = os.path.join(
        TRAIN_DIR,
        class_name,
        filename
    )

    # --------------------------------------------------------
    # Missing image
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        results.append({
            "isic_id": image_id,
            "target": target,
            "status": "missing",
            "width": None,
            "height": None,
            "brightness": None,
            "blur_score": None
        })

        corrupted += 1
        continue


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:

        results.append({
            "isic_id": image_id,
            "target": target,
            "status": "corrupted",
            "width": None,
            "height": None,
            "brightness": None,
            "blur_score": None
        })

        corrupted += 1
        continue


    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    height, width = image.shape[:2]

    size_problem = (
        width != EXPECTED_WIDTH
        or height != EXPECTED_HEIGHT
    )

    if size_problem:
        wrong_size += 1


    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Brightness
    # ============================================================
# ORIGINAL DATA MUST NOT BE MODIFIED
# This is a READ-ONLY quality check
# ============================================================
    # --------------------------------------------------------

    brightness = float(np.mean(gray))


    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


    # --------------------------------------------------------
    # Flags
    # --------------------------------------------------------

    dark_flag = brightness < DARK_THRESHOLD

    bright_flag = brightness > BRIGHT_THRESHOLD

    blur_flag = blur_score < BLUR_THRESHOLD


    if dark_flag:
        very_dark += 1

    if bright_flag:
        very_bright += 1

    if blur_flag:
        very_blurry += 1


    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    flags = []

    if size_problem:
        flags.append("wrong_size")

    if dark_flag:
        flags.append("very_dark")

    if bright_flag:
        flags.append("very_bright")

    if blur_flag:
        flags.append("very_blurry")


    if len(flags) == 0:
        status = "OK"
    else:
        status = ";".join(flags)


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({
        "isic_id": image_id,
        "target": target,
        "status": status,
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "blur_score": round(blur_score, 2)
    })


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (index + 1) % 1000 == 0:
        print(
            f"Checked {index + 1}/{len(df)} images..."
        )


# ============================================================
# SAVE REPORT
# ============================================================

report = pd.DataFrame(results)

report.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==========================================")
print("IMAGE QUALITY RESULTS")
print("==========================================")

print("Total checked:", len(report))

print("Corrupted/missing:", corrupted)

print("Wrong dimensions:", wrong_size)

print("Very dark:", very_dark)

print("Very bright:", very_bright)

print("Very blurry:", very_blurry)

print("\nImages with no flags:")

print(
    (report["status"] == "OK").sum()
)

print("\nQuality report saved to:")

print(OUTPUT_FILE)

print("\n==========================================")
print("IMPORTANT")
print("==========================================")

print(
    "No images were deleted or modified."
)
