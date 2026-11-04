import pandas as pd

# Path to metadata
metadata_path = r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset\train-metadata.csv"

df = pd.read_csv(metadata_path)

print("======================================")
print("ISIC 2019 PATIENT ANALYSIS")
print("======================================")

# Basic information
print("\nTotal images:", len(df))

print("Unique patients:", df["patient_id"].nunique())

print("Images per patient:")
print(df["patient_id"].value_counts().describe())

# Patients with multiple images
patient_counts = df["patient_id"].value_counts()

multiple_image_patients = patient_counts[patient_counts > 1]

print("\nPatients with more than 1 image:",
      len(multiple_image_patients))

print("\nTop patients by number of images:")
print(multiple_image_patients.head(20))

# Check missing patient IDs
print("\nMissing patient IDs:",
      df["patient_id"].isna().sum())

# Class distribution
print("\nClass distribution:")
print(df["target"].value_counts())

# Class distribution by patient
print("\n======================================")
print("CLASS DISTRIBUTION")
print("======================================")

print(
    df.groupby("target")["patient_id"]
      .nunique()
)

print("\nAnalysis completed.")
