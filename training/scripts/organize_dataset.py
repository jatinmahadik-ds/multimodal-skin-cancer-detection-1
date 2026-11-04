#!/usr/bin/env python3
"""
Organize ISIC 2019 dataset into cancer and non_cancer folders based on metadata.

This script reads the train-metadata.csv file and copies images from train-image/image/
into organized_dataset/cancer/ and organized_dataset/non_cancer/ based on the target column.

Target values:
    1 -> cancer
    0 -> non_cancer

The original dataset is NOT modified. Images are copied using shutil.copy2().
"""

import pandas as pd
import shutil
from pathlib import Path
import sys


def main():
    # Base directory
    base_dir = Path(r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset")
    
    # Input paths
    metadata_path = base_dir / "train-metadata.csv"
    images_dir = base_dir / "train-image" / "image"
    
    # Output paths
    output_dir = base_dir / "organized_dataset"
    cancer_dir = output_dir / "cancer"
    non_cancer_dir = output_dir / "non_cancer"
    
    # Create output directories
    cancer_dir.mkdir(parents=True, exist_ok=True)
    non_cancer_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify input paths exist
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found: {metadata_path}")
        sys.exit(1)
    
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Read metadata
    print("Reading metadata...")
    df = pd.read_csv(metadata_path)
    
    # Verify required columns
    required_cols = ['isic_id', 'target']
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' not found in metadata")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
    
    # Initialize counters
    total_rows = len(df)
    cancer_expected = 0
    non_cancer_expected = 0
    cancer_copied = 0
    non_cancer_copied = 0
    missing_images = []
    unexpected_targets = []
    
    print(f"Total metadata rows: {total_rows}")
    print("Processing images...")
    
    # Process each row
    for idx, row in df.iterrows():
        isic_id = row['isic_id']
        target = row['target']
        
        # Construct source image path
        # The isic_id already includes the full filename (including _downsampled suffix if present)
        src_path = images_dir / f"{isic_id}.jpg"
        
        # Check if source image exists
        if not src_path.exists():
            missing_images.append(f"{isic_id}.jpg")
            continue
        
        # Determine destination based on target
        if target == 1:
            cancer_expected += 1
            dst_path = cancer_dir / f"{isic_id}.jpg"
            try:
                shutil.copy2(src_path, dst_path)
                cancer_copied += 1
            except Exception as e:
                print(f"ERROR copying {isic_id}.jpg to cancer: {e}")
        elif target == 0:
            non_cancer_expected += 1
            dst_path = non_cancer_dir / f"{isic_id}.jpg"
            try:
                shutil.copy2(src_path, dst_path)
                non_cancer_copied += 1
            except Exception as e:
                print(f"ERROR copying {isic_id}.jpg to non_cancer: {e}")
        else:
            unexpected_targets.append((isic_id, target))
            continue
    
    # Verify final folder counts
    cancer_final_count = len(list(cancer_dir.glob("*.jpg")))
    non_cancer_final_count = len(list(non_cancer_dir.glob("*.jpg")))
    
    # Print detailed verification report
    print("\n" + "="*60)
    print("VERIFICATION REPORT")
    print("="*60)
    print(f"Total metadata rows:              {total_rows}")
    print(f"Cancer images expected:           {cancer_expected}")
    print(f"Non-cancer images expected:       {non_cancer_expected}")
    print(f"Cancer images successfully copied: {cancer_copied}")
    print(f"Non-cancer images successfully copied: {non_cancer_copied}")
    print(f"Missing images:                   {len(missing_images)}")
    print(f"Unexpected target values:         {len(unexpected_targets)}")
    print(f"Total images copied:              {cancer_copied + non_cancer_copied}")
    print("-"*60)
    print(f"Final cancer folder count:        {cancer_final_count}")
    print(f"Final non_cancer folder count:    {non_cancer_final_count}")
    print(f"Total in organized_dataset:       {cancer_final_count + non_cancer_final_count}")
    print("="*60)
    
    # Check for discrepancies
    print("\nDISCREPANCY ANALYSIS:")
    has_issues = False
    
    if cancer_copied != cancer_expected:
        print(f"  WARNING: Cancer copied ({cancer_copied}) != expected ({cancer_expected})")
        has_issues = True
    
    if non_cancer_copied != non_cancer_expected:
        print(f"  WARNING: Non-cancer copied ({non_cancer_copied}) != expected ({non_cancer_expected})")
        has_issues = True
    
    if cancer_final_count != cancer_copied:
        print(f"  WARNING: Cancer folder count ({cancer_final_count}) != copied ({cancer_copied})")
        has_issues = True
    
    if non_cancer_final_count != non_cancer_copied:
        print(f"  WARNING: Non-cancer folder count ({non_cancer_final_count}) != copied ({non_cancer_copied})")
        has_issues = True
    
    total_expected = cancer_expected + non_cancer_expected
    total_copied = cancer_copied + non_cancer_copied
    if total_copied != total_expected:
        print(f"  WARNING: Total copied ({total_copied}) != total expected ({total_expected})")
        has_issues = True
    
    if missing_images:
        print(f"  WARNING: {len(missing_images)} images missing from source")
        # Show first 10 missing
        for m in missing_images[:10]:
            print(f"    - {m}")
        if len(missing_images) > 10:
            print(f"    ... and {len(missing_images) - 10} more")
        has_issues = True
    
    if unexpected_targets:
        print(f"  WARNING: {len(unexpected_targets)} rows with unexpected target values")
        for isic_id, target in unexpected_targets[:10]:
            print(f"    - {isic_id}: target={target}")
        if len(unexpected_targets) > 10:
            print(f"    ... and {len(unexpected_targets) - 10} more")
        has_issues = True
    
    if not has_issues:
        print("  No discrepancies found. All counts match!")
    
    # Expected values from user
    print("\nEXPECTED VALUES (from user):")
    print(f"  Cancer: 4,522")
    print(f"  Non-cancer: 20,809")
    print(f"  Total: 25,331")
    
    if cancer_final_count != 4522:
        print(f"  NOTE: Cancer count differs by {cancer_final_count - 4522}")
    if non_cancer_final_count != 20809:
        print(f"  NOTE: Non-cancer count differs by {non_cancer_final_count - 20809}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()