# Clinical-Only XGBoost Artifacts

> **Naming note:** this folder follows the requested repository structure. No `fusion_xgboost.json` exists in the original project. The actual artifact is `clinical_xgboost.json`, a **clinical-only** XGBoost classifier. It does **not** consume EfficientNet probabilities and is not a multimodal fusion model.

## Files

| File | Purpose |
|---|---|
| `clinical_xgboost.json` | Trained XGBoost model, saved in XGBoost's JSON format. |
| `feature_schema.json` | Exact 53-feature order, raw field definitions, categorical vocabulary and missing-value policy. |
| `training_config.json` | Training hyperparameters, patient-grouped split strategy, threshold selection and missingness-analysis notes. |
| `clinical_features.py` | `ClinicalPreprocessor` used by the FastAPI backend to transform raw form fields into the 53-feature matrix. |
| `scripts/` | Original training, evaluation, inspection and prediction scripts. |
| `evaluation/` | Verified validation/test metrics, confusion matrix, feature importance and data-quality reports. |

## Verified model properties

| Property | Value |
|---|---|
| Algorithm | XGBoost `XGBClassifier` |
| Artifact size | 282,677 bytes |
| SHA-256 | `5EF9A01EFDBB31EA227EEAE930343256339C1952A8EE58440C594837ADDEB56A` |
| Training data | PAD-UFES-20 clinical metadata |
| Input features | 53 encoded clinical features |
| Image / CNN inputs | **None** |
| Target | Binary cancer / non-cancer, project-defined mapping |
| Seed | 42 |
| Max depth | 3 |
| Learning rate | 0.05 |
| Estimators | 2,000 with early stopping (50 rounds) |
| Split | Patient-grouped, stratified: 70% train / 15% validation / 15% test |

### Label mapping used for training

| PAD-UFES-20 diagnosis | Binary label |
|---|---|
| BCC, SCC, MEL | Cancer (1) |
| ACK, NEV, SEK | Non-cancer (0) |

This is a project-specific simplification. ACK is clinically significant and is not harmless simply because it is mapped to 0 here.

## Preprocessing and feature contract

- Raw form values are validated, upper-cased for categorical fields and matched against the saved vocabularies.
- Missing values are represented as `NaN`; XGBoost handles them natively. There is no imputation.
- `ClinicalPreprocessor` creates the exact 53 dummy/numeric columns in `feature_order`.
- The 21 raw clinical form fields are not the same as the 53 encoded model features.
- `age` is numeric; categorical vocabularies come from `feature_schema.json`.
- No `patient_id`, image path, diagnosis, `biopsed`, target or split column is passed to the model.

## Verified evaluation results

Evaluation files were produced by the original project on the held-out PAD-UFES-20 test split (343 records: 163 cancer, 180 non-cancer). Threshold `0.56` was selected on the **validation** split by maximizing cancer-class F1.

| Metric (test, threshold 0.56) | Value |
|---|---:|
| Accuracy | 0.8863 |
| Precision (cancer) | 0.8780 |
| Recall / sensitivity (cancer) | 0.8834 |
| Specificity | 0.8889 |
| F1 (cancer) | 0.8807 |
| Macro F1 | 0.8860 |
| ROC-AUC | 0.9596 |

At the default `0.50` threshold, the same test split reports accuracy 0.8921, precision 0.8706, recall 0.9080, specificity 0.8778, F1 0.8889 and ROC-AUC 0.9596.

### Inference threshold note

The FastAPI response's `assessment` label in `backend/prediction.py` uses a fixed `>= 0.5` display cut-off. The `0.56` value comes from the original validation-based evaluation and is recorded in `evaluation/metrics_test.json`. These are **uncalibrated model scores**, not clinical probabilities. See the caveats in that file, including the survey-absent subgroup being 100% non-cancer in this dataset.

## Training / evaluation scripts

Run these from this directory so `import clinical_features` resolves:

```powershell
python scripts/train_clinical_xgboost.py
python scripts/evaluate_clinical_xgboost.py
python scripts/predict_clinical.py
```

The scripts contain original absolute paths to `E:\Final year project\...`; update them before rerunning. The datasets are not included.

## No fusion model

`backend/model_loader.py` loads `clinical_xgboost.json` and the schema directly. It never loads a fusion model and never feeds EfficientNet output into XGBoost. If a validated fusion model is produced later, it should be added as a separate versioned artifact with its own schema, paired out-of-fold CNN predictions, evaluation report and explicit documentation.
