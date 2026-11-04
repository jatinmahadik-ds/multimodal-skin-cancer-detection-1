"""
schemas.py
================================================================================
Clinical input schema, form options and validation for the 3-model Doctor Portal.
Derived directly from XGboost/config/feature_schema.json.
================================================================================
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

CLINICAL_LABELS: dict[str, str] = {
    "age": "Age (years)",
    "diameter_1": "Lesion Diameter 1 (mm)",
    "diameter_2": "Lesion Diameter 2 (mm)",
    "fitspatrick": "Fitzpatrick Skin Type",
    "smoke": "Smoking",
    "drink": "Alcohol Consumption",
    "pesticide": "Pesticide Exposure",
    "skin_cancer_history": "Personal History of Skin Cancer",
    "cancer_history": "Personal History of Other Cancer",
    "has_piped_water": "Household Has Piped Water",
    "has_sewage_system": "Household Has Sewage System",
    "itch": "Itching",
    "grew": "Lesion Grew Recently",
    "hurt": "Pain",
    "changed": "Lesion Changed Recently",
    "bleed": "Bleeding",
    "elevation": "Raised / Elevated Lesion",
    "background_father": "Father's Ancestry",
    "background_mother": "Mother's Ancestry",
    "gender": "Gender",
    "region": "Lesion Body Region",
}

NUMERIC_SPEC: dict[str, dict[str, Any]] = {
    "age": {"integer": True, "min": 0, "max": 120, "required": True},
    "diameter_1": {"integer": False, "min": 0.0, "max": 500.0, "required": False},
    "diameter_2": {"integer": False, "min": 0.0, "max": 500.0, "required": False},
    "fitspatrick": {"integer": True, "min": 1, "max": 6, "required": False},
}

REQUIRED_CLINICAL = {"age", "region"}

MISSING_TOKENS = {"", "UNK", "NAN", "NONE", "NULL", "NA", "N/A", "-", "UNKNOWN"}


class ValidationError(Exception):
    def __init__(self, errors: list[dict[str, str]]):
        self.errors = errors
        super().__init__("Clinical validation failed")


def _format_option_label(feature: str, val: str) -> str:
    if feature == "fitspatrick":
        return f"Type {val}"
    if feature.startswith("background_"):
        return val.title()
    if val in ("TRUE", "FALSE"):
        return "Yes" if val == "TRUE" else "No"
    return val.replace("_", " ").title()


def clinical_options_payload(registry: Any) -> dict[str, Any]:
    """Build form descriptor exposed by GET /clinical-options."""
    numeric_list = []
    for name in registry.numeric_features:
        spec = NUMERIC_SPEC.get(name, {"integer": False, "min": 0.0, "max": 500.0, "required": False})
        numeric_list.append({
            "name": name,
            "label": CLINICAL_LABELS.get(name, name),
            "type": "integer" if spec["integer"] else "number",
            "min": spec["min"],
            "max": spec["max"],
            "required": name in REQUIRED_CLINICAL,
        })

    boolean_list = []
    for name in registry.boolean_features + registry.symptom_features:
        boolean_list.append({
            "name": name,
            "label": CLINICAL_LABELS.get(name, name),
            "type": "boolean",
            "required": name in REQUIRED_CLINICAL,
            "options": [
                {"value": "TRUE", "label": "Yes"},
                {"value": "FALSE", "label": "No"},
            ],
        })

    categorical_list = []
    for name in registry.categorical_features:
        vocab = registry.categorical_vocab.get(name, [])
        options = [{"value": v, "label": _format_option_label(name, v)} for v in sorted(vocab)]
        categorical_list.append({
            "name": name,
            "label": CLINICAL_LABELS.get(name, name),
            "type": "categorical",
            "required": name in REQUIRED_CLINICAL,
            "options": options,
        })

    return {
        "numeric_features": numeric_list,
        "boolean_features": boolean_list,
        "categorical_features": categorical_list,
        "required_fields": list(REQUIRED_CLINICAL),
    }


def validate_clinical(registry: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate raw incoming clinical strings from form."""
    errors: list[dict[str, str]] = []
    cleaned: dict[str, Any] = {}

    # Check required fields
    for req in REQUIRED_CLINICAL:
        v = values.get(req)
        if v is None or str(v).strip().upper() in MISSING_TOKENS:
            errors.append({"field": req, "message": f"'{CLINICAL_LABELS.get(req, req)}' is required."})

    # Validate numeric features
    for name in registry.numeric_features:
        v = values.get(name)
        if v is None:
            cleaned[name] = None
            continue
        sv = str(v).strip()
        if sv.upper() in MISSING_TOKENS:
            cleaned[name] = None
            continue

        spec = NUMERIC_SPEC.get(name, {"integer": False, "min": 0.0, "max": 500.0})
        try:
            num = int(sv) if spec["integer"] else float(sv)
        except ValueError:
            errors.append({"field": name, "message": f"Must be a valid {'integer' if spec['integer'] else 'number'}."})
            continue

        if num < spec["min"] or num > spec["max"]:
            errors.append({"field": name, "message": f"Must be between {spec['min']} and {spec['max']}."})
            continue
        cleaned[name] = num

    # Validate boolean & symptom features
    for name in registry.boolean_features + registry.symptom_features:
        v = values.get(name)
        if v is None:
            cleaned[name] = None
            continue
        sv = str(v).strip().upper()
        if sv in MISSING_TOKENS:
            cleaned[name] = None
            continue
        if sv in ("TRUE", "YES", "1", "T", "Y"):
            cleaned[name] = "TRUE"
        elif sv in ("FALSE", "NO", "0", "F", "N"):
            cleaned[name] = "FALSE"
        else:
            errors.append({"field": name, "message": "Must be Yes or No."})

    # Validate categorical features
    for name in registry.categorical_features:
        v = values.get(name)
        if v is None:
            cleaned[name] = None
            continue
        sv = str(v).strip().upper()
        if sv in MISSING_TOKENS:
            cleaned[name] = None
            continue
        vocab = registry.categorical_vocab.get(name, [])
        if vocab and sv not in vocab:
            errors.append({"field": name, "message": f"Unknown category '{sv}'. Must be one of {vocab}."})
            continue
        cleaned[name] = sv

    if errors:
        raise ValidationError(errors)

    return cleaned


# ---------------------------------------------------------------------------
# Pydantic Request Models for FastAPI
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field


class PatientCreate(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., min_length=1)
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    patient_id: Optional[str] = None


class PatientUpdate(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., min_length=1)
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None


class ReportCreate(BaseModel):
    patient_id: str
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    patient_phone: Optional[str] = None
    patient_email: Optional[str] = None
    efficientnet_score: float
    resnet50_score: Optional[float] = None
    clinical_score: float
    ensemble_score: Optional[float] = None
    weights: Optional[dict] = None
    threshold: Optional[float] = None
    prediction: Optional[int] = None
    prediction_label: Optional[str] = None
    risk_level: Optional[str] = None
    clinical_inputs: Optional[dict] = None
    doctor_notes: Optional[str] = ""
    dual_model_status: Optional[str] = "Independent models; no unvalidated ensemble"


class DoctorLogin(BaseModel):
    email: str
    password: str


class DoctorSignup(BaseModel):
    fullName: str
    regNumber: str
    email: str
    phone: str
    specialization: str
    hospital: str
    password: str
