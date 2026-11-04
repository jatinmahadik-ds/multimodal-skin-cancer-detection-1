# EfficientNetB0 Model Artifacts

## Weight file (not committed)

The trained EfficientNetB0 artifact is intentionally excluded from Git because it is a binary model weight file.

| Property | Value |
|---|---|
| Expected filename | `efficientnet_b0_best.keras` |
| Expected location | `training/models/efficientnet_b0_best.keras` |
| Size | 28,956,745 bytes (≈27.6 MiB / 28.9 MB) |
| SHA-256 | `6FD74D96F47446E10581C9E7BF775E4385362C4124C68B44B763BBDC73A9558D` |
| Framework | TensorFlow 2.19.0 / Keras 3.15.1 |
| Input | 224 × 224 × 3 RGB image |
| Output | Single sigmoid cancer score |
| Training dataset | ISIC 2019 (224×224 resized working set) |

### How to place the model

The repository does not host or link to a public copy of the trained weights. Obtain the verified artifact from the project author or the project's private/institutional artifact storage, verify the SHA-256 above, then place it at:

```
training/models/efficientnet_b0_best.keras
```

The backend resolves this path relative to the repository root. To load the weights from a different directory without copying them, set:

```powershell
$env:SKIN_MODELS_DIR = "D:\path\to\model\directory"
```

The file must be named `efficientnet_b0_best.keras` inside that directory.

### Do not substitute another weight file

The reported evaluation results belong to the exact artifact identified by the SHA-256 above. Replacing it with a newly trained or externally downloaded model invalidates those results.

## Verified evaluation outputs

| File | Contents |
|---|---|
| `evaluation/test_metrics.csv` | Held-out ISIC 2019 test metrics |
| `evaluation/classification_report.txt` | Per-class precision, recall, F1 and support |
| `evaluation/confusion_matrix.png` | Confusion matrix from the test run |
| `evaluation/training_history.csv` | Recorded training history |

Verified test-set results (2,522 images):

| Metric | Value |
|---|---:|
| Accuracy | 0.8049 |
| Precision (cancer) | 0.4740 |
| Recall / sensitivity (cancer) | 0.8053 |
| F1 (cancer) | 0.5967 |
| ROC-AUC | 0.8956 |

## ResNet50

ResNet50 weights are **not** used by the final architecture and are not documented as an active model. Any historical ResNet50 `.keras` files are excluded by `.gitignore`.
