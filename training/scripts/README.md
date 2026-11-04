# Image Pipeline and Dataset Scripts

These scripts are the actual ISIC 2019 image-preparation and EfficientNetB0 source files from the original project. They are kept for reproducibility and academic review.

## Warning

- The scripts contain the **original absolute Windows paths** (`E:\Final year project\...`). Update them before running on another machine.
- Several scripts can **move, overwrite, rebalance or delete files** in the dataset directory. Run them only against a disposable copy of the dataset. This repository does not include the dataset.
- The final architecture does not use ResNet50. ResNet50 training scripts were intentionally excluded.

## Files

| Script | Purpose |
|---|---|
| `organize_dataset.py` | Organizes ISIC 2019 images into cancer / non-cancer folders using `train-metadata.csv`. |
| `split_dataset.py` | Creates train / validation / test splits. |
| `balance_training.py` | Balances the training manifest/class distribution. |
| `regenerate_balanced_train.py` | Regenerates `balanced_train.csv` from the cleaned training split by oversampling the cancer class in the CSV only. |
| `remove_duplicates.py` | Removes duplicate images identified by hashing. |
| `duplicate_check.py` | Hashes images and reports duplicate groups. |
| `duplicate_inspect.py` | Inspects duplicate groups in detail. |
| `duplicate_report.py` | Produces a duplicate report across splits. |
| `check_quality.py` | Checks image readability, dimensions, channels and related quality issues. |
| `inspect_quality.py` | Summarizes dataset quality with tables/plots. |
| `verify_cleaned_dataset.py` | Read-only duplicate verification of the cleaned split. |
| `final_verification.py` | Read-only full ISIC 2019 verification (decode, size, RGB, hashes, manifest consistency). |
| `patient_analysis.py` | Prints ISIC 2019 metadata/patient-level analysis. |
| `efficientnet_training.py` | Final EfficientNetB0 ISIC 2019 training script (two-phase head training + fine-tuning). |
| `predict_efficientnet.py` | Standalone EfficientNetB0 inference helper. |

## Original training configuration

The final EfficientNetB0 script records:

- Backbone: EfficientNetB0, ImageNet initialization, no original top
- Input: 224 × 224 × 3
- Head: global average pooling → dropout → single sigmoid unit
- Batch size: 16
- Head phase: 10 epochs, learning rate `1e-4`
- Fine-tuning phase: 10 epochs, learning rate `1e-5`, final 30 layers unfrozen
- Seed: 42
- Saved artifacts: `efficientnet_b0_best.keras`, `efficientnet_b0_final.keras`

The best-weight artifact is not committed. See `../models/README.md`.
