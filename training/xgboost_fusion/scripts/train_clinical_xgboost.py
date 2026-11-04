"""
train_clinical_xgboost.py
=========================
Leakage-aware training pipeline for a clinical-only XGBoost binary classifier
on PAD-UFES-20 (metadata.csv only - no images, no CNN outputs).

Pipeline steps:
  1. Load metadata, map the project binary target (rejecting unknown labels).
  2. Patient-grouped, row-level-stratified 70/15/15 split (seed 42).
  3. Fit the clinical preprocessor on TRAIN only; transform all splits.
  4. Train XGBoost with early stopping on the VALIDATION split.
  5. Select the decision threshold on VALIDATION only (0.5 reported as reference).
  6. Save model, feature schema, split CSVs, integrity report, validation
     metrics, feature importance, and the missingness-confounding analysis
     (including a documented sensitivity experiment on the survey-present
     sub-cohort).

Usage:
    "<venv python>" train_clinical_xgboost.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

import clinical_features as cf
from metrics_utils import compute_binary_metrics, select_threshold

ROOT = Path(__file__).resolve().parents[1]
METADATA = Path(r"E:\Final year project\PAD-UFES-20\metadata.csv")
CONFIG = ROOT / "training_config.json"
DATA = ROOT / "data"
MODELS = ROOT
REPORTS = ROOT / "evaluation"
CONFIG_OUT = ROOT / "feature_schema.json"

SEED = 42


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def pkg_versions() -> dict:
    import xgboost, sklearn, scipy, matplotlib
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "xgboost": xgboost.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


# ---------------------------------------------------------------------------
# 1. Data loading & target
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(METADATA)
    log(f"loaded metadata: {df.shape[0]} rows x {df.shape[1]} cols")
    df, target_report = cf.add_target_and_analysis_columns(df)
    if target_report["unknown_diagnoses"]:
        raise ValueError(f"Unmapped diagnoses rejected: {target_report['unknown_diagnoses']}")
    counts = df["cancer_target"].value_counts().to_dict()
    log(f"class counts after project mapping: non-cancer(0)={counts.get(0,0)} cancer(1)={counts.get(1,0)}")
    return df


# ---------------------------------------------------------------------------
# 2. Leakage-safe splitting (grouped by patient, stratified at row level)
# ---------------------------------------------------------------------------

def make_splits(df: pd.DataFrame) -> dict:
    """
    StratifiedGroupKFold over 20 folds with patient_id as the group.
    Each patient lands in exactly ONE fold, so grouping is guaranteed; we then
    allocate folds 0-2 -> test (~15%), 3-5 -> validation (~15%), 6-19 -> train
    (~70%). Row-level stratification keeps class balance close to the target
    even though 110 patients carry mixed-label lesions (patient-level majority
    stratification would distort those patients' true label distribution).
    """
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    y = df["cancer_target"].values
    groups = df["patient_id"].values
    fold_ids = np.full(len(df), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(df)), y, groups)):
        fold_ids[test_idx] = fold

    test_folds, val_folds = {0, 1, 2}, {3, 4, 5}
    split_of_row = np.where(
        np.isin(fold_ids, list(test_folds)), "test",
        np.where(np.isin(fold_ids, list(val_folds)), "validation", "train"),
    )
    return {
        "train": df[split_of_row == "train"].reset_index(drop=True),
        "validation": df[split_of_row == "validation"].reset_index(drop=True),
        "test": df[split_of_row == "test"].reset_index(drop=True),
        "fold_ids": fold_ids,
    }


def verify_and_report_splits(df: pd.DataFrame, splits: dict) -> dict:
    sets = {k: set(splits[k]["patient_id"]) for k in ("train", "validation", "test")}
    overlap = {
        "train_val": sorted(sets["train"] & sets["validation"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "validation_test": sorted(sets["validation"] & sets["test"]),
    }
    all_overlap_zero = all(len(v) == 0 for v in overlap.values())

    def class_block(s: pd.DataFrame) -> dict:
        n = len(s)
        n1 = int(s["cancer_target"].sum())
        n0 = n - n1
        return {
            "rows": n,
            "cancer_1": n1,
            "non_cancer_0": n0,
            "cancer_rate": round(n1 / n, 4) if n else None,
            "patients": int(s["patient_id"].nunique()),
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "method": (
            "StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42) with "
            "patient_id groups; folds 0-2=test, 3-5=validation, 6-19=train. Each "
            "patient belongs to exactly one fold by construction, so no patient "
            "can appear in two splits."
        ),
        "class_counts": {k: class_block(splits[k]) for k in ("train", "validation", "test")},
        "patient_overlap": {k: {"n_overlapping_patients": len(v), "patients": v[:50]} for k, v in overlap.items()},
        "patient_overlap_zero": all_overlap_zero,
        "patient_overlap_all_patients_listed_if_small": True,
        "lesion_id_note": "lesion_id is NOT used for grouping: 250 lesion ids are reused across different patients in PAD-UFES-20.",
        "test_set_policy": "test folds were excluded from preprocessing fits, early stopping, and threshold selection",
    }
    if not all_overlap_zero:
        raise RuntimeError(f"Patient overlap between splits detected: {[k for k,v in overlap.items() if v]}")
    for k, v in report["class_counts"].items():
        log(f"split {k:10s} rows={v['rows']:5d} patients={v['patients']:5d} cancer_rate={v['cancer_rate']}")
    return report


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> XGBClassifier:
    m = dict(cfg["model"])
    early = m.pop("early_stopping_rounds")
    m.pop("notes", None)
    m.pop("random_state", None)  # enforced below from the global seed
    return XGBClassifier(
        **m,
        early_stopping_rounds=early,
        random_state=SEED,
    )


def train_model(cfg: dict, X_tr, y_tr, X_val, y_val):
    model = build_model(cfg)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    log(f"best_iteration={model.best_iteration} best_val_logloss={model.best_score:.5f}")
    return model


# ---------------------------------------------------------------------------
# 4. Missingness confounding analysis
# ---------------------------------------------------------------------------

def missingness_analysis(cfg: dict, splits: dict, model, X_val, y_val, threshold: float) -> dict:
    """
    Investigate the known PAD-UFES-20 missingness concern: a block of survey
    fields is absent for ~35% of records, and those records are exclusively
    non-cancer diagnoses. The indicator is analysis-only (NOT a model feature).
    """
    out: dict = {
        "indicator_definition": {
            "name": "survey_block_present",
            "fields": cf.SURVEY_BLOCK_FIELDS,
            "used_as_model_feature": False,
            "note": "defined for analysis only; the model never receives this indicator explicitly (it can still indirectly exploit per-feature missingness, which is exactly what we are measuring)",
        },
        "target_by_survey_presence_per_split": {},
        "validation_subgroup_performance": {},
    }

    val_scores = model.predict_proba(X_val)[:, 1]

    for name in ("train", "validation", "test"):
        s = splits[name]
        block = {}
        for present_val, label in ((True, "survey_present"), (False, "survey_absent")):
            sub = s[s["survey_block_present"] == present_val]
            n = len(sub)
            n1 = int(sub["cancer_target"].sum())
            block[label] = {
                "rows": n,
                "cancer_1": n1,
                "non_cancer_0": n - n1,
                "cancer_rate": round(n1 / n, 4) if n else None,
            }
        out["target_by_survey_presence_per_split"][name] = block

    # Subgroup performance on the validation split at the chosen threshold.
    present_mask = splits["validation"]["survey_block_present"].values
    for label, mask in (("survey_present", present_mask), ("survey_absent", ~present_mask)):
        if mask.sum() == 0:
            out["validation_subgroup_performance"][label] = {"n": 0, "metrics": None,
                "note": "no records in this subgroup on validation"}
            continue
        out["validation_subgroup_performance"][label] = {
            "n": int(mask.sum()),
            "metrics": compute_binary_metrics(y_val[mask], val_scores[mask], threshold),
        }

    # Trivial majority baseline on validation (overall and by subgroup).
    out["majority_baseline_validation"] = compute_binary_metrics(
        y_val, np.full(len(y_val), float(np.mean(y_val))), 0.5
    )["majority_class_baseline"]

    # ---- sensitivity experiment: refit on survey-present sub-cohort only ----
    out["sensitivity_experiment"] = run_sensitivity_experiment(cfg, splits)
    return out


def run_sensitivity_experiment(cfg: dict, splits: dict) -> dict:
    """
    Exploratory, documented sensitivity check:

    Refit the identical pipeline (same hyperparameters, same seed, same
    patient-grouped splits) using ONLY records where the survey block is
    present, then evaluate on the survey-present subset of the test split.
    This keeps patients grouped (splits are inherited from the primary
    pipeline) and does not influence the primary model or its threshold.
    """
    sub = {k: splits[k][splits[k]["survey_block_present"]].reset_index(drop=True)
           for k in ("train", "validation", "test")}
    pre = cf.ClinicalPreprocessor().fit(sub["train"])
    X_tr, _ = pre.transform(sub["train"], stage="sensitivity_train")
    X_va, _ = pre.transform(sub["validation"], stage="sensitivity_validation")
    X_te, _ = pre.transform(sub["test"], stage="sensitivity_test")
    y_tr = sub["train"]["cancer_target"].values
    y_va = sub["validation"]["cancer_target"].values
    y_te = sub["test"]["cancer_target"].values

    model = train_model(cfg, X_tr, y_tr, X_va, y_va)
    thr = select_threshold(y_va, model.predict_proba(X_va)[:, 1])
    scores_te = model.predict_proba(X_te)[:, 1]

    return {
        "description": (
            "Refit of the identical pipeline on the survey-present sub-cohort only "
            "(patients stay grouped; the primary model and its threshold are untouched). "
            "EXPLORATORY - not the primary reported model."
        ),
        "sub_cohort_sizes": {k: int(len(v)) for k, v in sub.items()},
        "sub_cohort_class_counts": {k: {"cancer_1": int(v['cancer_target'].sum()),
                                        "non_cancer_0": int(len(v) - v['cancer_target'].sum())}
                                    for k, v in sub.items()},
        "best_iteration": int(model.best_iteration),
        "validation_threshold_selection": thr,
        "test_metrics_at_validation_threshold": compute_binary_metrics(y_te, scores_te, thr["threshold"]),
        "test_metrics_at_default_0.5": compute_binary_metrics(y_te, scores_te, 0.5),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for d in (DATA, MODELS, REPORTS, CONFIG.parent):
        d.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    versions = pkg_versions()
    log(f"versions: {versions}")

    # 1. data ---------------------------------------------------------------
    df = load_data()

    # 2. splits -------------------------------------------------------------
    splits = make_splits(df)
    integrity = verify_and_report_splits(df, splits)
    (REPORTS / "split_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    for name in ("train", "validation", "test"):
        splits[name].to_csv(DATA / f"{name}.csv", index=False)
    log(f"saved split CSVs to {DATA}")

    # 3. preprocessing (FIT ON TRAIN ONLY) ----------------------------------
    pre = cf.ClinicalPreprocessor().fit(splits["train"])
    X = {}
    transform_reports = {}
    for name in ("train", "validation", "test"):
        X[name], rep = pre.transform(splits[name], stage=name)
        transform_reports[name] = rep
        log(f"transform {name}: X shape={X[name].shape}")
    for name, rep in transform_reports.items():
        if rep["unexpected_tokens"]:
            log(f"  unexpected tokens in {name}: {rep['unexpected_tokens']}")
        if rep["unseen_categories"]:
            log(f"  unseen categories in {name}: {rep['unseen_categories']}")

    y = {name: splits[name]["cancer_target"].values for name in ("train", "validation", "test")}

    # 4. train --------------------------------------------------------------
    model = train_model(cfg, X["train"], y["train"], X["validation"], y["validation"])

    # 5. threshold on VALIDATION only ---------------------------------------
    val_scores = model.predict_proba(X["validation"])[:, 1]
    thr_info = select_threshold(y["validation"], val_scores)
    threshold = float(thr_info["threshold"])
    log(f"validation-selected threshold={threshold} (reference 0.5)")

    val_metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "clinical-only XGBoost (no images, no CNN outputs)",
        "best_iteration": int(model.best_iteration),
        "best_validation_logloss": float(model.best_score),
        "threshold_selection": thr_info,
        "metrics_at_selected_threshold": compute_binary_metrics(y["validation"], val_scores, threshold),
        "metrics_at_default_0.5": compute_binary_metrics(y["validation"], val_scores, 0.5),
        "test_set_policy": "the test split was NOT used for training, early stopping, threshold selection or preprocessing fits",
        "versions": versions,
    }
    (REPORTS / "metrics_validation.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

    # 6. save model ---------------------------------------------------------
    model_path = MODELS / "clinical_xgboost.json"
    model.save_model(str(model_path))
    log(f"saved model to {model_path}")

    # 7. feature schema -----------------------------------------------------
    schema = {
        **pre.to_schema(),
        "project": "clinical-only XGBoost on PAD-UFES-20",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            "name": "cancer_target",
            "positive_class": 1,
            "negative_class": 0,
            "mapping": {"cancer_1": sorted(cf.CANCER_CLASSES), "non_cancer_0": sorted(cf.NON_CANCER_CLASSES)},
            "note": ("PROJECT-SPECIFIC binary label definition, not a universal clinical "
                     "definition. ACK (actinic keratosis) can be a precancerous lesion and "
                     "is grouped with the negative class only for this project."),
        },
        "excluded_columns": cf.EXCLUDED_COLUMNS,
        "missing_tokens": sorted(cf.MISSING_TOKENS),
        "missing_value_policy": "empty strings and 'UNK' are missing; missing values stay NaN and are handled natively by XGBoost (no imputation)",
        "fitpatrick_policy": "ordinal 1-6; values normalised via numeric conversion so 1, 1.0 and '1.0' are identical; out-of-range values become missing",
        "seed": SEED,
        "n_training_rows_used_to_fit_categories": int(len(splits["train"])),
    }
    CONFIG_OUT.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    log(f"saved feature schema ({len(pre.feature_order)} features) to {CONFIG_OUT}")

    # 8. feature importance -------------------------------------------------
    imp = model.get_booster().get_score(importance_type="gain")
    rows = sorted(
        ({"feature": f, "gain": imp.get(f, 0.0)} for f in pre.feature_order),
        key=lambda r: -r["gain"],
    )
    total = sum(r["gain"] for r in rows) or 1.0
    for r in rows:
        r["gain_pct"] = round(100.0 * r["gain"] / total, 4)
    pd.DataFrame(rows).to_csv(REPORTS / "feature_importance.csv", index=False)
    log(f"top-5 gain features: {[r['feature'] for r in rows[:5]]}")

    # 9. missingness confounding analysis -----------------------------------
    miss_report = missingness_analysis(cfg, splits, model, X["validation"], y["validation"], threshold)
    miss_report["interpretation_guidance"] = (
        "If the survey-absent subgroup is 100% non-cancer, any overall test score "
        "partially reflects dataset collection patterns (missingness -> non-cancer) "
        "rather than lesion-clinical signal. Compare the survey-present subgroup and "
        "the sensitivity experiment before drawing conclusions; do not claim clinical "
        "generalisation from a high overall score alone."
    )
    (REPORTS / "missingness_analysis.json").write_text(json.dumps(miss_report, indent=2), encoding="utf-8")
    log("saved missingness analysis")

    log("training pipeline complete")


if __name__ == "__main__":
    main()
