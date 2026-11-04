"""
inspect_dataset.py
==================
Dataset inspection report for the clinical-only XGBoost project on PAD-UFES-20.

Reads metadata.csv, checks the environment, and writes a machine-readable
report to reports/dataset_inspection.json covering:

  * shape, column names, dtypes,
  * missing-value counts and percentages,
  * unique `diagnostic` values and mapping status,
  * duplicate records and identifier checks (patient_id / lesion_id / img_id),
  * class counts before and after the project binary mapping,
  * patient counts and records per patient,
  * missingness patterns in the clinical survey block (including its
    association with the diagnosis - the known confounding concern).

Usage:
    "<venv python>" inspect_dataset.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import clinical_features as cf

ROOT = Path(__file__).resolve().parents[1]
METADATA = Path(r"E:\Final year project\PAD-UFES-20\metadata.csv")
REPORTS = ROOT / "evaluation"


def py_version() -> str:
    return sys.version.split()[0]


def pkg_versions() -> dict:
    out = {}
    for name in ("xgboost", "scikit-learn", "pandas", "numpy", "matplotlib", "scipy"):
        try:
            mod = __import__(name.replace("-", "_") if name != "scikit-learn" else "sklearn")
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover
            out[name] = f"NOT AVAILABLE ({exc})"
    return out


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA)
    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_path": str(METADATA),
        "environment": {
            "python": py_version(),
            "platform": platform.platform(),
            "packages": pkg_versions(),
        },
    }

    # -- shape & dtypes ----------------------------------------------------
    report["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    report["columns"] = [
        {"name": c, "dtype": str(df[c].dtype), "n_unique": int(df[c].nunique(dropna=True))}
        for c in df.columns
    ]

    # -- missing values ----------------------------------------------------
    miss = df.isna().sum()
    empty_str = {c: int((df[c].astype(str).str.strip() == "").sum()) for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"}
    unk_str = {c: int((df[c].astype(str).str.strip().str.upper() == "UNK").sum()) for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"}
    report["missing_values"] = {
        c: {
            "nan_count": int(miss[c]),
            "nan_pct": round(100.0 * miss[c] / len(df), 2),
            "empty_string_count": empty_str.get(c, 0),
            "literal_UNK_count": unk_str.get(c, 0),
            "effective_missing_pct": round(
                100.0 * (miss[c] + empty_str.get(c, 0) + unk_str.get(c, 0)) / len(df), 2
            ),
        }
        for c in df.columns
    }

    # -- diagnoses ---------------------------------------------------------
    diag_counts = df["diagnostic"].value_counts(dropna=False)
    mapping_status, unknown = {}, {}
    for d, n in diag_counts.items():
        y, known = cf.normalize_target(d)
        mapping_status[str(d)] = {"count": int(n), "mapped": known, "target": y}
        if not known:
            unknown[str(d)] = int(n)
    report["diagnostics"] = {
        "unique_values": sorted(str(v) for v in diag_counts.index),
        "value_counts": {str(k): int(v) for k, v in diag_counts.items()},
        "mapping": mapping_status,
        "unmapped_values": unknown,
        "mapping_definition": (
            "PROJECT-SPECIFIC binary label definition, not a universal clinical "
            "definition. Cancer=1: BCC, SCC, MEL. Non-cancer=0: ACK, NEV, SEK. "
            "ACK (actinic keratosis) can represent a precancerous lesion and is "
            "grouped with the negative class only for this project's binary task."
        ),
    }

    # -- class counts before/after mapping ---------------------------------
    targets, _ = zip(*[cf.normalize_target(d) for d in df["diagnostic"]])
    y = pd.Series(targets, dtype="int64")
    report["class_counts"] = {
        "before_mapping": {str(k): int(v) for k, v in diag_counts.items()},
        "after_mapping": {"non_cancer_0": int((y == 0).sum()), "cancer_1": int((y == 1).sum())},
        "positive_rate": round(float(y.mean()), 4),
    }

    # -- duplicates & identifier checks ------------------------------------
    per_patient = df.groupby("patient_id").size()
    lesion_shared = df.groupby("lesion_id")["patient_id"].nunique()
    report["identifiers"] = {
        "duplicate_full_rows": int(df.duplicated().sum()),
        "n_unique_patient_id": int(df["patient_id"].nunique()),
        "n_unique_lesion_id": int(df["lesion_id"].nunique()),
        "n_unique_img_id": int(df["img_id"].nunique()),
        "img_id_unique_per_row": bool(df["img_id"].nunique() == len(df)),
        "lesion_ids_shared_across_patients": int((lesion_shared > 1).sum()),
        "lesion_id_globally_unique": bool(lesion_shared.max() == 1),
        "grouping_key": "patient_id (lesion_id alone is NOT safe: 250 lesion ids are reused across different patients)",
        "records_per_patient": {
            "min": int(per_patient.min()),
            "max": int(per_patient.max()),
            "mean": round(float(per_patient.mean()), 3),
            "median": float(per_patient.median()),
            "patients_with_multiple_records": int((per_patient > 1).sum()),
        },
    }

    # -- survey-block missingness ------------------------------------------
    block = cf.SURVEY_BLOCK_FIELDS
    present = cf.survey_block_present(df)
    per_field = {f: int(df[f].isna().sum()) for f in block}
    ct = pd.crosstab(df["diagnostic"], present)
    ct_dict = {
        str(idx): {"survey_absent": int(row.get(False, 0)), "survey_present": int(row.get(True, 0))}
        for idx, row in ct.iterrows()
    }
    report["survey_block_missingness"] = {
        "fields": block,
        "rows_with_survey_absent": int((~present).sum()),
        "rows_with_survey_present": int(present.sum()),
        "nan_per_field": per_field,
        "diagnosis_by_survey_presence": ct_dict,
        "concern": (
            "Every record missing the survey block carries a non-cancer diagnosis "
            "(ACK/NEV/SEK); all cancer records (BCC/MEL/SCC) have the block present. "
            "A model can therefore exploit missingness as a proxy for the negative "
            "class. This is investigated explicitly in training/evaluation."
        ),
    }

    # -- symptom-field UNK pattern -----------------------------------------
    report["symptom_unk_counts"] = {
        f: int((df[f].astype(str).str.strip().str.upper() == "UNK").sum())
        for f in cf.SYMPTOM_SOURCES
    }

    # -- biopsed leakage check (documented, never used as a feature) -------
    report["biopsed_leakage_check"] = {
        "diagnostic_x_biopsed": {
            str(idx): {"not_biopsed": int(row.get(False, 0)), "biopsed": int(row.get(True, 0))}
            for idx, row in pd.crosstab(df["diagnostic"], df["biopsed"]).iterrows()
        },
        "reason_excluded": cf.EXCLUDED_COLUMNS["biopsed"],
    }

    out = REPORTS / "dataset_inspection.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[inspect] wrote {out}")
    print(f"[inspect] rows={report['shape']['rows']} cols={report['shape']['columns']}")
    print(f"[inspect] classes after mapping: {report['class_counts']['after_mapping']}")
    print(f"[inspect] unmapped diagnoses: {unknown or 'none'}")
    print(f"[inspect] survey-block absent rows: {report['survey_block_missingness']['rows_with_survey_absent']}")


if __name__ == "__main__":
    main()
