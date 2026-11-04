"""
clinical_features.py
====================
Shared, leakage-aware feature engineering for the clinical-only XGBoost model
trained on PAD-UFES-20 metadata.

This module is used by training, evaluation AND inference so that all three
stages share exactly the same:

  * target (binary label) definition,
  * raw column -> model feature mapping,
  * missing-value normalisation rules,
  * categorical encoding (categories are FIT ON TRAINING DATA ONLY and then
    frozen in the feature schema JSON for reuse at inference time),
  * final ordered feature list.

Design rules enforced here (see README.md for the full rationale):

  * Clinical columns only. Identifiers (patient_id, lesion_id, img_id), the
    diagnosis itself, and `biopsed` (a post-diagnostic process variable that
    leaks target information) are NEVER model inputs.
  * Empty strings and the literal token "UNK" are treated as missing.
  * Missing values are preserved as NaN so XGBoost's native missing-value
    handling can be used. Nothing is imputed with 0 or with a "missing"
    category.
  * Categorical dummy columns are defined from the TRAINING split only.
    Unseen categories at validation/test/inference time are mapped to an
    all-zero dummy block ("observed but unseen"), never silently dropped.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

SEED = 42

# Tokens treated as "missing" in any textual/boolean/categorical field.
MISSING_TOKENS = {"", "UNK", "UNKNOWN", "NONE", "NAN", "NA", "N/A", "-", "?"}

# --- Project-specific binary label definition (NOT a universal clinical one) --
# Cancer = 1 : BCC, SCC, MEL
# Non-cancer = 0 : ACK, NEV, SEK
# NOTE: ACK (actinic keratosis) can represent a precancerous lesion. Grouping
# it with the negative class is a project decision made for this binary task.
CANCER_CLASSES = {"BCC", "SCC", "MEL"}
NON_CANCER_CLASSES = {"ACK", "NEV", "SEK"}

# --- Raw metadata columns used as model inputs (verified against metadata.csv)
NUMERIC_SOURCES = {
    "age": "age",
    "diameter_1": "diameter_1",
    "diameter_2": "diameter_2",
    "fitspatrick": "fitspatrick",  # ordinal skin type 1..6
}
BOOLEAN_SOURCES = {
    "smoke": "smoke",
    "drink": "drink",
    "pesticide": "pesticide",
    "skin_cancer_history": "skin_cancer_history",
    "cancer_history": "cancer_history",
    "has_piped_water": "has_piped_water",
    "has_sewage_system": "has_sewage_system",
}
SYMPTOM_SOURCES = {
    "itch": "itch",
    "grew": "grew",
    "hurt": "hurt",
    "changed": "changed",
    "bleed": "bleed",
    "elevation": "elevation",
}
CATEGORICAL_SOURCES = {
    "background_father": "background_father",
    "background_mother": "background_mother",
    "gender": "gender",
    "region": "region",
}

# Columns explicitly excluded from the feature matrix, with reasons.
EXCLUDED_COLUMNS = {
    "patient_id": "Patient identifier - grouping key for leakage-safe splits; never a predictor.",
    "lesion_id": "Lesion identifier - not globally unique (reused across patients) and is an identifier, not a clinical predictor.",
    "img_id": "Image filename - image-derived; this model is clinical-only.",
    "diagnostic": "Raw diagnosis label - direct source of the target.",
    "biopsed": "Post-diagnostic biopsy decision - leaks information about the diagnostic process and the target (all BCC/MEL/SCC lesions were biopsied).",
}

# The 11-field survey block whose absence is strongly associated with
# non-cancer diagnoses in PAD-UFES-20. Used for MISSINGNESS ANALYSIS ONLY;
# it is deliberately NOT a model feature.
SURVEY_BLOCK_FIELDS = [
    "smoke", "drink", "pesticide", "gender", "skin_cancer_history",
    "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick",
    "diameter_1", "diameter_2",
]

FITSpatrick_MIN, FITSpatrick_MAX = 1.0, 6.0

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _is_missing_scalar(v: Any) -> bool:
    """True for None / NaN / missing-token strings."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(v, str) and v.strip().upper() in MISSING_TOKENS:
        return True
    return False


def normalize_diagnostic(value: Any) -> Optional[str]:
    """Upper-case, stripped diagnosis string, or None if missing/unrecognised."""
    if _is_missing_scalar(value):
        return None
    s = str(value).strip().upper()
    return s if s else None


def normalize_target(diagnostic: Any) -> Tuple[Optional[int], bool]:
    """
    Map a raw diagnostic value to the project binary label.

    Returns (target, is_known). Unknown diagnoses are NEVER silently mapped
    to class 0; they return (None, False) so callers can reject them.
    """
    d = normalize_diagnostic(diagnostic)
    if d in CANCER_CLASSES:
        return 1, True
    if d in NON_CANCER_CLASSES:
        return 0, True
    return None, False


def to_bool01(value: Any) -> Tuple[float, Optional[str]]:
    """
    Convert messy boolean representations to 1.0 / 0.0 / NaN.

    Returns (value, unexpected_token_or_None).
    """
    if _is_missing_scalar(value):
        return np.nan, None
    if isinstance(value, (bool, np.bool_)):
        return (1.0 if bool(value) else 0.0), None
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return float(value), None
        return np.nan, f"int_out_of_range:{value}"
    if isinstance(value, (float, np.floating)):
        if float(value) in (0.0, 1.0):
            return float(value), None
        return np.nan, f"float_out_of_range:{value}"
    s = str(value).strip().upper()
    if s in MISSING_TOKENS:
        return np.nan, None
    if s in {"TRUE", "T", "YES", "Y", "1"}:
        return 1.0, None
    if s in {"FALSE", "F", "NO", "N", "0"}:
        return 0.0, None
    return np.nan, f"unexpected_bool_token:{s}"


def to_numeric(value: Any) -> Tuple[float, Optional[str]]:
    """Convert to float, reporting (rather than hiding) conversion failures."""
    if _is_missing_scalar(value):
        return np.nan, None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        v = float(value)
        return (np.nan, "non_finite") if not math.isfinite(v) else (v, None)
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan, f"numeric_coercion_failed:{value!r}"
    return (np.nan, "non_finite") if not math.isfinite(v) else (v, None)


def normalize_category(value: Any, field: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalise a categorical value to an upper-case string (or None if missing).

    Field-specific fixes (verified in PAD-UFES-20):
      * background_father/background_mother: "BRASIL" -> "BRAZIL".
      * gender: "F"/"M" are expanded for safety at inference time.
    """
    if _is_missing_scalar(value):
        return None, None
    s = " ".join(str(value).strip().upper().split())
    if s in MISSING_TOKENS:
        return None, None
    if field in ("background_father", "background_mother") and s == "BRASIL":
        s = "BRAZIL"  # same country, two spellings in the dataset
    if field == "gender":
        s = {"F": "FEMALE", "M": "MALE"}.get(s, s)
    return s, None


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class ClinicalPreprocessor:
    """
    Clinical-only preprocessor for XGBoost.

    * fit() learns categorical vocabularies from TRAINING data only.
    * transform() returns a float64 DataFrame with the exact schema feature
      order. Missing values stay NaN (native XGBoost missing handling).
    * The full state serialises to a dict, which is stored in
      config/feature_schema.json so inference reproduces training exactly.
    """

    VERSION = "1.0"

    def __init__(self) -> None:
        self.categories: Dict[str, List[str]] = {}
        self.feature_order: List[str] = []
        self._is_fitted = False

    # -- fit ---------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "ClinicalPreprocessor":
        self.categories = {}
        for field in sorted(CATEGORICAL_SOURCES):
            if field not in df.columns:
                raise KeyError(f"Expected categorical column {field!r} not found.")
            seen = set()
            for v in df[field].tolist():
                cat, _ = normalize_category(v, field)
                if cat is not None:
                    seen.add(cat)
            self.categories[field] = sorted(seen)
        self.feature_order = self._build_feature_order()
        self._is_fitted = True
        return self

    def _build_feature_order(self) -> List[str]:
        order: List[str] = []
        order.extend(NUMERIC_SOURCES)          # age, diameter_1, diameter_2, fitspatrick
        order.extend(BOOLEAN_SOURCES)          # survey yes/no flags
        order.extend(SYMPTOM_SOURCES)          # itch, grew, hurt, changed, bleed, elevation
        for field in sorted(CATEGORICAL_SOURCES):
            for cat in self.categories.get(field, []):
                order.append(f"{field}={cat}")
        return order

    # -- transform ---------------------------------------------------------
    def transform(self, df: pd.DataFrame, stage: str = "transform") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not self._is_fitted:
            raise RuntimeError("ClinicalPreprocessor must be fitted (or loaded from schema) first.")

        report: Dict[str, Any] = {
            "stage": stage,
            "n_rows": int(len(df)),
            "unexpected_tokens": {},
            "unseen_categories": {},
        }

        def note(key: str, token: Optional[str]) -> None:
            if token is not None:
                bucket = report["unexpected_tokens"]
                bucket.setdefault(key, {})
                bucket[key][token] = bucket[key].get(token, 0) + 1

        X = pd.DataFrame(index=df.index)

        # Numeric: age, diameter_1, diameter_2, fitspatrick ---------------
        for name, src in NUMERIC_SOURCES.items():
            vals, fails = [], []
            for v in df[src].tolist():
                num, err = to_numeric(v)
                if err is not None:
                    fails.append(err)
                vals.append(num)
            if name == "fitspatrick":
                cleaned = []
                for num in vals:
                    if not np.isnan(num) and not (FITSpatrick_MIN <= num <= FITSpatrick_MAX):
                        cleaned.append(np.nan)  # out-of-range skin type -> missing
                        note(name, "out_of_range_value")
                    else:
                        cleaned.append(num)
                vals = cleaned
            if fails:
                report["unexpected_tokens"].setdefault(name, {})
                for f in fails:
                    report["unexpected_tokens"][name][f] = report["unexpected_tokens"][name].get(f, 0) + 1
            X[name] = pd.array(vals, dtype="float64")

        # Booleans and symptoms: 1.0 / 0.0 / NaN ---------------------------
        for name in list(BOOLEAN_SOURCES) + list(SYMPTOM_SOURCES):
            src = name
            vals = []
            for v in df[src].tolist():
                b, err = to_bool01(v)
                note(name, err)
                vals.append(b)
            X[name] = pd.array(vals, dtype="float64")

        # Categoricals: training-fitted dummy encoding ---------------------
        for field in sorted(CATEGORICAL_SOURCES):
            cats = self.categories[field]
            normed: List[Optional[str]] = []
            unseen_counter: Dict[str, int] = {}
            for v in df[field].tolist():
                cat, _ = normalize_category(v, field)
                if cat is not None and cat not in cats:
                    unseen_counter[cat] = unseen_counter.get(cat, 0) + 1
                normed.append(cat)
            if unseen_counter:
                report["unseen_categories"][field] = unseen_counter
            s = pd.Series(normed, index=df.index, dtype="object")
            for cat in cats:
                # Missing source -> NaN in every dummy (native XGBoost missing
                # handling). Observed-but-unseen -> all-zero dummy block.
                X[f"{field}={cat}"] = pd.Series(
                    np.where(s.isna(), np.nan, (s == cat).astype(float)),
                    index=df.index, dtype="float64",
                )

        X = X[self.feature_order]
        return X.astype("float64"), report

    # -- serialisation -----------------------------------------------------
    def to_schema(self) -> Dict[str, Any]:
        return {
            "schema_version": self.VERSION,
            "numeric_features": dict(NUMERIC_SOURCES),
            "boolean_features": dict(BOOLEAN_SOURCES),
            "symptom_features": dict(SYMPTOM_SOURCES),
            "categorical_features": {
                field: {
                    "source_column": CATEGORICAL_SOURCES[field],
                    "categories_from_training": self.categories[field],
                    "dummy_columns": [f"{field}={c}" for c in self.categories[field]],
                    "missing_policy": "all dummies set to NaN when the source value is missing",
                    "unseen_policy": "all dummies set to 0 (observed but not seen during training)",
                }
                for field in sorted(CATEGORICAL_SOURCES)
            },
            "feature_order": list(self.feature_order),
        }

    @classmethod
    def from_schema(cls, schema: Dict[str, Any]) -> "ClinicalPreprocessor":
        pre = cls()
        pre.categories = {
            field: list(spec["categories_from_training"])
            for field, spec in schema["categorical_features"].items()
        }
        pre.feature_order = list(schema["feature_order"])
        pre._is_fitted = True
        return pre


def survey_block_present(df: pd.DataFrame) -> pd.Series:
    """
    Analysis-only indicator: does a record have ANY value in the 11-field
    survey block? (In PAD-UFES-20 this block is missing for ~35% of records,
    and those records are exclusively non-cancer diagnoses.)
    """
    return df[SURVEY_BLOCK_FIELDS].notna().any(axis=1)


def add_target_and_analysis_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Add `cancer_target` and `survey_block_present` columns for split files.
    Raises if any diagnosis cannot be mapped - unknown labels are never
    silently assigned to class 0.
    """
    targets, unknowns = [], {}
    for v in df["diagnostic"].tolist():
        y, known = normalize_target(v)
        if not known:
            key = str(v)
            unknowns[key] = unknowns.get(key, 0) + 1
        targets.append(y)
    if unknowns:
        raise ValueError(
            "Unmapped diagnosis values found (refusing to train): "
            f"{unknowns}. Update the project label mapping explicitly if intended."
        )
    out = df.copy()
    out["cancer_target"] = pd.array(targets, dtype="int64")
    out["survey_block_present"] = survey_block_present(df).values
    return out, {"unknown_diagnoses": unknowns}
