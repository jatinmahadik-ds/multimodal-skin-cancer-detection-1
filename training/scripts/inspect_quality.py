import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset"

REPORT = os.path.join(
    BASE_DIR,
    "dataset_split",
    "image_quality_report.csv"
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset_split",
    "train"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset_split",
    "quality_samples"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD REPORT
# ============================================================

df = pd.read_csv(REPORT)

# Separate classes
cancer_blurry = df[
    (df["target"] == 1) &
    (df["status"].str.contains("very_blurry", na=False))
]

non_cancer_blurry = df[
    (df["target"] == 0) &
    (df["status"].str.contains("very_blurry", na=False))
]

# Normal examples
cancer_ok = df[
    (df["target"] == 1) &
    (df["status"] == "OK")
]

non_cancer_ok = df[
    (df["target"] == 0) &
    (df["status"] == "OK")
]


# ============================================================
# SAMPLE
# ============================================================

cancer_blurry = cancer_blurry.sample(
    min(20, len(cancer_blurry)),
    random_state=42
)

non_cancer_blurry = non_cancer_blurry.sample(
    min(20, len(non_cancer_blurry)),
    random_state=42
)

cancer_ok = cancer_ok.sample(
    min(10, len(cancer_ok)),
    random_state=42
)

non_cancer_ok = non_cancer_ok.sample(
    min(10, len(non_cancer_ok)),
    random_state=42
)


# ============================================================
# CONTACT SHEET FUNCTION
# ============================================================

def create_contact_sheet(data, class_name, filename):

    if len(data) == 0:
        print("No images found for:", class_name)
        return

    columns = 5
    rows = (len(data) + columns - 1) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 3 * rows)
    )

    # Make axes iterable
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, data.iterrows()):

        image_id = str(row["isic_id"])

        image_path = os.path.join(
            TRAIN_DIR,
            class_name,
            image_id + ".jpg"
        )

        try:

            image = plt.imread(image_path)

            ax.imshow(image)

            ax.set_title(
                f"{image_id}\nBlur: {row['blur_score']:.1f}",
                fontsize=8
            )

            ax.axis("off")

        except Exception as e:

            ax.set_title(
                f"Error\n{image_id}",
                fontsize=8
            )

            ax.axis("off")

    # Hide unused axes
    for ax in axes[len(data):]:
        ax.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


# ============================================================
# CREATE CONTACT SHEETS
# ============================================================

create_contact_sheet(
    cancer_blurry,
    "cancer",
    "cancer_blurry_samples.jpg"
)

create_contact_sheet(
    non_cancer_blurry,
    "non_cancer",
    "non_cancer_blurry_samples.jpg"
)

create_contact_sheet(
    cancer_ok,
    "cancer",
    "cancer_normal_samples.jpg"
)

create_contact_sheet(
    non_cancer_ok,
    "non_cancer",
    "non_cancer_normal_samples.jpg"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==========================================")
print("INSPECTION COMPLETE")
print("==========================================")

print("Cancer blurry samples:", len(cancer_blurry))
print("Non-cancer blurry samples:", len(non_cancer_blurry))

print("\nImages saved in:")
print(OUTPUT_DIR)
