# Training and Model Artifacts

This folder documents the training source and verified evaluation artifacts. Datasets and large model weights are intentionally **not** committed.

## Final active architecture

The active inference pipeline contains two independent models:

1. **EfficientNetB0** — binary skin-lesion image classifier trained on ISIC 2019.
2. **Clinical-only XGBoost** — tabular classifier trained on PAD-UFES-20 clinical metadata (53 encoded features).

The two outputs are returned separately. No validated fusion score is produced. ResNet50 has been removed from the final architecture.

## Layout

| Path | Purpose |
|---|---|
| `scripts/` | Image dataset preparation, cleaning, splitting, EfficientNetB0 training, evaluation and prediction utilities. |
| `models/` | Drop-in location for `efficientnet_b0_best.keras` (not committed) and verified EfficientNetB0 evaluation outputs. |
| `xgboost_fusion/` | Actual XGBoost artifact, feature schema, training configuration, clinical preprocessor, pipeline scripts and evaluation reports. |

## Important naming note

The folder is named `xgboost_fusion/` to match the requested repository structure. However, **no `fusion_xgboost.json` was found** in the original project, and the active model is not a fusion model. The real artifact is:

```
training/xgboost_fusion/clinical_xgboost.json
```

This is the **clinical-only** XGBoost classifier. It consumes 53 clinical dummy features and never receives CNN probabilities. See `xgboost_fusion/README.md`.

## Datasets

The datasets are not included. Training scripts expect the original local dataset layout and contain the original absolute Windows paths. If you re-run them, update those paths first.

| Dataset | Used for | Included here |
|---|---|---|
| ISIC 2019 (224×224 resized) | EfficientNetB0 image classifier | No — obtain from the official ISIC archive |
| PAD-UFES-20 metadata | Clinical-only XGBoost | No — obtain from the official PAD-UFES-20 source |
| HAM10000 | Historical ResNet50 branch only | No — not used by the final active pipeline |

## Verified metrics

Only the metrics present in the original evaluation files were copied. See:

- `models/evaluation/` for EfficientNetB0 test metrics and confusion matrix.
- `xgboost_fusion/evaluation/` for XGBoost validation/test metrics, confusion matrix, feature importance and missingness reports.

No metric was re-estimated, rounded up, or invented.

## Reproducibility

The Environment used for the backend and verification was Python 3.12.10 with the pinned versions in `backend/requirements.txt`. TensorFlow/Keras and XGBoost versions are sensitive to the saved artifacts; do not upgrade them without re-validating model loading and predictions.
