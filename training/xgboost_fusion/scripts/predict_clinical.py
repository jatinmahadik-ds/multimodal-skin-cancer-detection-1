"""
predict_clinical.py
===================
Reproducible inference for the clinical-only XGBoost model.

  * Loads models/clinical_xgboost.json and config/feature_schema.json.
  * Accepts CLINICAL FIELDS ONLY - never requires a skin image or any
    CNN/image-derived prediction.
  * Validates required inputs, safely handles optional missing values
    (empty strings / "UNK" / None are treated as missing).
  * Uses the exact training feature order from the feature schema.
  * Returns the cancer-class score, the non-cancer-class score and the
    chosen binary prediction (validation-selected threshold).

IMPORTANT: the returned scores are XGBoost discriminative outputs, NOT
calibrated medical probabilities. Calibration has not been evaluated.

Usage:
    "<venv python>" predict_clinical.py --example
    "<venv python>" predict_clinical.py --json "{\"age\": 65, ...}"
    "<venv python>" predict_clinical.py --json-file record.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import clinical_features as cf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "clinical_xgboost.json"
SCHEMA_PATH = ROOT / "config" / "feature_schema.json"
VAL_METRICS_PATH = ROOT / "reports" / "metrics_validation.json"

# Every clinical input the model understands (source column names).
CLINICAL_INPUT_FIELDS = (
    list(cf.NUMERIC_SOURCES)
    + list(cf.BOOLEAN_SOURCES)
    + list(cf.SYMPTOM_SOURCES)
    + list(cf.CATEGORICAL_SOURCES)
)

_model = None
_pre = None
_threshold = 0.5


def load_artifacts() -> tuple:
    """Load model, preprocessor (from schema) and validation-selected threshold."""
    global _model, _pre, _threshold
    if _model is not None:
        return _model, _pre, _threshold
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train the model first.")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Feature schema not found: {SCHEMA_PATH}.")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    _model = XGBClassifier()
    _model.load_model(str(MODEL_PATH))
    _pre = cf.ClinicalPreprocessor.from_schema(schema)
    if VAL_METRICS_PATH.exists():
        val = json.loads(VAL_METRICS_PATH.read_text(encoding="utf-8"))
        _threshold = float(val["threshold_selection"]["threshold"])
    return _model, _pre, _threshold


def predict_clinical_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a single clinical record.

    Parameters
    ----------
    record : dict
        Clinical fields only, e.g. {"age": 65, "gender": "MALE", "itch": "TRUE",
        "region": "FACE", ...}. Fields may be omitted or set to None/""/"UNK"
        for missing values.

    Returns
    -------
    dict with cancer_score, non_cancer_score, prediction, threshold, and a
    list of input warnings (missing required fields, unknown categories, etc.).
    """
    model, pre, threshold = load_artifacts()

    if not isinstance(record, dict):
        raise TypeError("record must be a dict of clinical fields")

    warnings = []
    # Reject fields this model does not know (protects against accidental
    # image/CNN inputs or typos being silently ignored).
    unknown_fields = sorted(set(record) - set(CLINICAL_INPUT_FIELDS))
    if unknown_fields:
        warnings.append(
            f"ignored unknown input fields: {unknown_fields}. This model accepts "
            f"clinical fields only: {CLINICAL_INPUT_FIELDS}"
        )

    # `age` is the only truly required field in PAD-UFES-20 (0% missing).
    if record.get("age") is None or (isinstance(record.get("age"), float) and np.isnan(record.get("age"))):
        warnings.append("required field 'age' is missing; the score will rely on other fields")

    row = {f: record.get(f) for f in CLINICAL_INPUT_FIELDS}
    df = pd.DataFrame([row])
    X, rep = pre.transform(df, stage="inference")

    if rep["unexpected_tokens"]:
        warnings.append(f"values could not be interpreted and were treated as missing: {rep['unexpected_tokens']}")
    if rep["unseen_categories"]:
        warnings.append(f"categories not seen during training encoded as all-zero dummies: {rep['unseen_categories']}")
    n_missing = int(X.isna().sum().sum())
    if n_missing:
        warnings.append(f"{n_missing} feature values are missing (handled natively by XGBoost)")

    p_cancer = float(model.predict_proba(X.values if not hasattr(X, "columns") else X)[:, 1][0])
    prediction = int(p_cancer >= threshold)
    return {
        "cancer_score": p_cancer,
        "non_cancer_score": 1.0 - p_cancer,
        "prediction": prediction,
        "prediction_label": "cancer" if prediction == 1 else "non-cancer",
        "threshold": threshold,
        "score_disclaimer": (
            "These are XGBoost discriminative scores, NOT calibrated medical "
            "probabilities. Calibration has not been evaluated."
        ),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# SYNTHETIC demonstration record - NOT a real patient.
# ---------------------------------------------------------------------------
SYNTHETIC_EXAMPLE = {
    "_note": "SYNTHETIC RECORD FOR API DEMONSTRATION ONLY - not an actual patient.",
    "age": 68,
    "gender": "MALE",
    "region": "FACE",
    "itch": "TRUE",
    "grew": "TRUE",
    "hurt": "FALSE",
    "changed": "TRUE",
    "bleed": "TRUE",
    "elevation": "TRUE",
    "smoke": "FALSE",
    "drink": "FALSE",
    "pesticide": "TRUE",
    "skin_cancer_history": "FALSE",
    "cancer_history": "TRUE",
    "has_piped_water": "TRUE",
    "has_sewage_system": "TRUE",
    "fitspatrick": 2,
    "diameter_1": 9.0,
    "diameter_2": 7.5,
    "background_father": "GERMANY",
    "background_mother": "POMERANIA",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Clinical-only XGBoost inference on PAD-UFES-20 features")
    ap.add_argument("--example", action="store_true", help="run the built-in SYNTHETIC example record")
    ap.add_argument("--json", type=str, help="a JSON string with clinical fields")
    ap.add_argument("--json-file", type=str, help="path to a JSON file with clinical fields")
    args = ap.parse_args()

    if args.example:
        record = {k: v for k, v in SYNTHETIC_EXAMPLE.items() if not k.startswith("_")}
        print("Running SYNTHETIC example record (NOT a real patient):")
        print(json.dumps(SYNTHETIC_EXAMPLE, indent=2))
    elif args.json:
        record = json.loads(args.json)
    elif args.json_file:
        record = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    else:
        ap.error("provide --example, --json or --json-file")

    result = predict_clinical_record(record)
    print("\nPrediction result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
