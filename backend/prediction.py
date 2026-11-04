"""
prediction.py
================================================================================
Independent inference workflow for the two verified models:

    1. User uploads one skin lesion image.
    2. User enters required clinical data.
    3. EfficientNetB0 processes the image (224x224 RGB, ISIC 2019 preprocessing).
    4. Clinical-only XGBoost processes clinical data only (53 dummy features from feature_schema.json).
       No CNN predictions or image features are passed to Model 2.
    5. Cancer-class score is obtained from each model independently.
    6. No artificial or unvalidated 2-model ensemble is calculated.
       EfficientNetB0 and Clinical XGBoost outputs are returned as separate,
       independent assessments.
================================================================================
"""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

IMG_SIZE = 224
MAX_IMAGE_PIXELS = 50_000_000
ALLOWED_FORMATS = {"JPEG", "PNG"}


class ImageValidationError(Exception):
    pass


class InferenceError(Exception):
    pass


def validate_image_bytes(data: bytes) -> str:
    """Validate uploaded bytes are genuine, non-corrupt JPEG or PNG."""
    if not data:
        raise ImageValidationError("The uploaded image is empty.")

    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except UnidentifiedImageError as exc:
        raise ImageValidationError("The file is not a readable image (corrupt or unsupported format).") from exc
    except Exception as exc:
        raise ImageValidationError("The image could not be decoded (corrupt or truncated).") from exc

    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError(f"Unsupported format '{fmt}'. Please upload JPG or PNG.")

    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception as exc:
        raise ImageValidationError("The image could not be reopened for validation.") from exc

    if w * h > MAX_IMAGE_PIXELS:
        raise ImageValidationError(f"Image too large ({w}x{h} pixels). Maximum is {MAX_IMAGE_PIXELS} pixels.")

    return fmt


def load_image_rgb(data: bytes) -> np.ndarray:
    """Load image bytes into a 224x224x3 float32 RGB numpy array in range [0, 255]."""
    with Image.open(io.BytesIO(data)) as im:
        rgb_im = im.convert("RGB")
        resized = rgb_im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return np.array(resized, dtype=np.float32)


def run_prediction(registry: Any, image_bytes: bytes, clinical_record: Dict[str, Any]) -> Dict[str, Any]:
    """Execute dual-model independent inference (EfficientNetB0 + Clinical XGBoost).
    
    ResNet50 is not loaded or executed.
    No unvalidated 2-model ensemble is created.
    """
    from tensorflow.keras.applications import efficientnet

    with registry.inference_lock():
        # 1. Process Image for Model 1: EfficientNetB0
        try:
            raw_rgb = load_image_rgb(image_bytes)
            eff_batch = np.expand_dims(raw_rgb.copy(), axis=0)
            eff_in = efficientnet.preprocess_input(eff_batch)
            eff_pred = registry.efficientnet.predict(eff_in, verbose=0)
            eff_cancer_score = float(np.asarray(eff_pred).reshape(-1)[0])
            eff_cancer_score = max(0.0, min(1.0, eff_cancer_score))
        except Exception as exc:
            raise InferenceError(f"EfficientNetB0 inference failed: {exc}") from exc

        # 2. Process Clinical Data for Model 2: Clinical-Only XGBoost
        try:
            # Prepare row containing only clinical fields (no CNN outputs or image features)
            import clinical_features as cf
            all_clinical_fields = (
                list(cf.NUMERIC_SOURCES)
                + list(cf.BOOLEAN_SOURCES)
                + list(cf.SYMPTOM_SOURCES)
                + list(cf.CATEGORICAL_SOURCES)
            )
            clean_row = {f: clinical_record.get(f) for f in all_clinical_fields}
            df_in = pd.DataFrame([clean_row])

            X, rep = registry.clinical_preprocessor.transform(df_in, stage="inference")
            feat_order = registry.feature_order
            X_ordered = X[feat_order]

            cli_probs = registry.clinical_xgboost.predict_proba(X_ordered)[0]
            cli_cancer_score = float(cli_probs[1])
            cli_cancer_score = max(0.0, min(1.0, cli_cancer_score))
        except Exception as exc:
            raise InferenceError(f"Clinical XGBoost inference failed: {exc}") from exc

        timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "prediction_id": f"PRED-{timestamp[:10].replace('-', '')}-{int(datetime.now().timestamp())}-{uuid.uuid4().hex[:6].upper()}",
            "created_at": timestamp,
            "pipeline": "dual_model_independent",
            "models": {
                "efficientnet_b0": {
                    "model_name": "EfficientNetB0 (ISIC 2019)",
                    "modality": "Dermoscopic Lesion Image (224x224 RGB)",
                    "cancer_score": eff_cancer_score,
                    "non_cancer_score": 1.0 - eff_cancer_score,
                    "assessment": "Cancer Pattern Suspected" if eff_cancer_score >= 0.5 else "Benign Lesion Pattern",
                },
                "clinical_xgboost": {
                    "model_name": "Clinical-Only XGBoost (PAD-UFES-20)",
                    "modality": "Patient Demographics, History & Symptoms",
                    "cancer_score": cli_cancer_score,
                    "non_cancer_score": 1.0 - cli_cancer_score,
                    "assessment": "High-Risk Clinical Profile" if cli_cancer_score >= 0.5 else "Low-Risk Clinical Profile",
                    "features_used": len(feat_order),
                },
            },
            "dual_model_status": {
                "ensemble_available": False,
                "notice": (
                    "Independent dual-model outputs. EfficientNetB0 evaluates skin lesion imagery alone, "
                    "while Clinical-Only XGBoost evaluates demographic and symptom features alone. "
                    "Because an ensemble combining these two models has not been statistically validated "
                    "on paired test data, no combined ensemble score or final merged classification is generated."
                ),
            },
            "clinical_inputs": clinical_record,
            "score_disclaimer": "These scores are model estimates and are not calibrated medical diagnostic probabilities.",
        }
