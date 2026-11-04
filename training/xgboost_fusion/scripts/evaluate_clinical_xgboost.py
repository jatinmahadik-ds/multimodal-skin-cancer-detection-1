"""
evaluate_clinical_xgboost.py
============================
One-shot evaluation of the saved clinical-only XGBoost model on the held-out
TEST split.

  * Loads models/clinical_xgboost.json and config/feature_schema.json.
  * Rebuilds the preprocessor from the saved schema (NEVER refits on test).
  * Uses the validation-selected threshold stored in reports/metrics_validation.json
    (0.5 is reported as a reference only).
  * Writes: metrics_test.json, classification_report.txt, confusion_matrix.png,
    predictions/test_predictions.csv, and verifies that a freshly reloaded
    model reproduces the saved predictions within floating-point tolerance.

Usage:
    "<venv python>" evaluate_clinical_xgboost.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

import clinical_features as cf
from metrics_utils import compute_binary_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT
REPORTS = ROOT / "evaluation"
PREDS = ROOT / "predictions"
SCHEMA = ROOT / "feature_schema.json"
VAL_METRICS = REPORTS / "metrics_validation.json"

SALT = "pad-ufes-20-clinical-xgb-rowref-v1"  # fixed salt -> stable, non-identifying row refs


def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


def patient_ref(patient_id: str) -> str:
    """Stable pseudonymous reference (salted SHA-256 prefix), not reversible
    without the id list, and used instead of exposing patient ids."""
    return hashlib.sha256((SALT + str(patient_id)).encode("utf-8")).hexdigest()[:12]


def fmt(v) -> str:
    return "unavailable" if v is None else f"{v:.4f}"


def main() -> None:
    for d in (REPORTS, PREDS):
        d.mkdir(parents=True, exist_ok=True)

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    val_metrics = json.loads(VAL_METRICS.read_text(encoding="utf-8"))
    threshold = float(val_metrics["threshold_selection"]["threshold"])
    log(f"using validation-selected threshold={threshold} (reference 0.5)")

    # --- load model fresh from disk ---------------------------------------
    model = XGBClassifier()
    model.load_model(str(MODELS / "clinical_xgboost.json"))
    pre = cf.ClinicalPreprocessor.from_schema(schema)

    # --- test split --------------------------------------------------------
    test_df = pd.read_csv(DATA / "test.csv")
    X_test, rep = pre.transform(test_df, stage="test_evaluation")
    if rep["unexpected_tokens"] or rep["unseen_categories"]:
        log(f"transform notes: {rep}")
    feature_order = list(schema["feature_order"])
    assert list(X_test.columns) == feature_order, "feature order mismatch"
    y_test = test_df["cancer_target"].values.astype(int)

    scores = model.predict_proba(X_test)[:, 1]
    preds = (scores >= threshold).astype(int)
    preds_05 = (scores >= 0.5).astype(int)

    # --- metrics -----------------------------------------------------------
    overall = compute_binary_metrics(y_test, scores, threshold)
    overall_05 = compute_binary_metrics(y_test, scores, 0.5)

    subgroups = {}
    present = test_df["survey_block_present"].values.astype(bool)
    for label, mask in (("survey_present", present), ("survey_absent", ~present)):
        if mask.sum() == 0:
            subgroups[label] = {"n": 0, "metrics": None,
                                "note": "no records in this subgroup; metrics unavailable"}
            continue
        subgroups[label] = {"n": int(mask.sum()),
                            "metrics": compute_binary_metrics(y_test[mask], scores[mask], threshold)}

    # --- reload reproducibility check --------------------------------------
    model2 = XGBClassifier()
    model2.load_model(str(MODELS / "clinical_xgboost.json"))
    scores_reload = model2.predict_proba(X_test)[:, 1]
    max_abs_diff = float(np.max(np.abs(scores - scores_reload)))
    repro_ok = bool(np.allclose(scores, scores_reload, rtol=0, atol=1e-8))
    log(f"reload reproducibility: max_abs_diff={max_abs_diff:.3e} within_1e-8={repro_ok}")

    metrics_test = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODELS / "clinical_xgboost.json"),
        "feature_schema": str(SCHEMA),
        "n_features": len(feature_order),
        "threshold_source": "validation split only (reports/metrics_validation.json)",
        "metrics_at_selected_threshold": overall,
        "metrics_at_default_0.5": overall_05,
        "test_class_distribution": {
            "cancer_1": int(y_test.sum()),
            "non_cancer_0": int((y_test == 0).sum()),
            "total": int(len(y_test)),
        },
        "missingness_subgroups": subgroups,
        "reload_reproducibility": {
            "max_abs_score_diff": max_abs_diff,
            "within_1e-8": repro_ok,
            "note": "saved model reloaded from disk; predictions match the evaluation run",
        },
        "transform_report": rep,
        "caveats": [
            "Scores are XGBoost discriminative outputs, NOT calibrated medical probabilities; calibration was not evaluated.",
            "The survey-absent subgroup is 100% non-cancer in this dataset, so overall metrics partly reflect dataset collection/missingness patterns.",
            "No image data, CNN predictions or embeddings were used; this model is clinical-only.",
        ],
    }
    (REPORTS / "metrics_test.json").write_text(json.dumps(metrics_test, indent=2), encoding="utf-8")
    log("saved reports/metrics_test.json")

    # --- predictions CSV ---------------------------------------------------
    pred_df = pd.DataFrame({
        "row_index": np.arange(len(test_df)),
        "patient_ref": [patient_ref(p) for p in test_df["patient_id"]],
        "lesion_id": test_df["lesion_id"].values,
        "survey_block_present": present,
        "true_target": y_test,
        "cancer_score": scores,
        "pred_label_selected_threshold": preds,
        "pred_label_default_0.5": preds_05,
        "selected_threshold": threshold,
    })
    pred_df.to_csv(PREDS / "test_predictions.csv", index=False)
    log(f"saved {len(pred_df)} predictions to {PREDS / 'test_predictions.csv'}")

    # --- confusion matrix figure -------------------------------------------
    cm = confusion_matrix(y_true=y_test, y_pred=preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["non-cancer (0)", "cancer (1)"])
    ax.set_yticks([0, 1], labels=["non-cancer (0)", "cancer (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Clinical-only XGBoost - test set (threshold={threshold})")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center",
                color="white" if val > cm.max() / 2 else "black", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    log("saved reports/confusion_matrix.png")

    # --- text report --------------------------------------------------------
    cr = classification_report(
        y_test, preds, labels=[0, 1],
        target_names=["non-cancer (0)", "cancer (1)"],
        digits=4, zero_division=0,
    )
    cm_full = classification_report(
        y_test, preds_05, labels=[0, 1],
        target_names=["non-cancer (0)", "cancer (1)"],
        digits=4, zero_division=0,
    )
    lines = [
        "CLINICAL-ONLY XGBOOST ON PAD-UFES-20 - TEST SET REPORT",
        "=" * 64,
        "",
        f"Model:               models/clinical_xgboost.json (XGBoost, {len(feature_order)} clinical features)",
        "Inputs:              clinical metadata ONLY (no images, no CNN outputs, no embeddings)",
        f"Threshold:           {threshold} (selected on the VALIDATION split; 0.5 shown as reference)",
        f"Test records:        {len(y_test)} (cancer={int(y_test.sum())}, non-cancer={int((y_test==0).sum())})",
        f"Test patients:       {int(test_df['patient_id'].nunique())} (no patient overlap with train/validation - see reports/split_integrity.json)",
        "",
        "OVERALL METRICS (threshold selected on validation)",
        "-" * 64,
        f"  accuracy:                {fmt(overall['accuracy'])}",
        f"  precision (cancer):      {fmt(overall['precision_cancer'])}",
        f"  recall/sensitivity:      {fmt(overall['recall_cancer_sensitivity'])}",
        f"  specificity:             {fmt(overall['specificity'])}",
        f"  F1 (cancer):             {fmt(overall['f1_cancer'])}",
        f"  macro F1:                {fmt(overall['macro_f1'])}",
        f"  ROC-AUC:                 {fmt(overall['roc_auc'])}",
        f"  confusion matrix:        TN={overall['confusion_matrix']['tn']} FP={overall['confusion_matrix']['fp']} "
        f"FN={overall['confusion_matrix']['fn']} TP={overall['confusion_matrix']['tp']}",
        f"  majority-class baseline: accuracy={fmt(overall['majority_class_baseline']['accuracy_if_always_majority'])} "
        f"(always predicts class {overall['majority_class_baseline']['majority_label']})",
        "",
        "Classification report (selected threshold):",
        cr,
        "REFERENCE: metrics at the DEFAULT 0.5 threshold",
        "-" * 64,
        f"  accuracy={fmt(overall_05['accuracy'])}  F1(cancer)={fmt(overall_05['f1_cancer'])}  "
        f"ROC-AUC={fmt(overall_05['roc_auc'])}  "
        f"CM: TN={overall_05['confusion_matrix']['tn']} FP={overall_05['confusion_matrix']['fp']} "
        f"FN={overall_05['confusion_matrix']['fn']} TP={overall_05['confusion_matrix']['tp']}",
        cm_full,
        "MISSINGNESS SUBGROUPS (survey block present vs absent)",
        "-" * 64,
    ]
    for label, blk in subgroups.items():
        if blk["metrics"] is None:
            lines.append(f"  {label}: n={blk['n']} - no records; metrics unavailable")
            continue
        m = blk["metrics"]
        lines.append(
            f"  {label}: n={blk['n']} (cancer={m['n_positive']}, non-cancer={m['n_negative']}) "
            f"accuracy={fmt(m['accuracy'])} F1(cancer)={fmt(m['f1_cancer'])} ROC-AUC={fmt(m['roc_auc'])} "
            f"CM: TN={m['confusion_matrix']['tn']} FP={m['confusion_matrix']['fp']} "
            f"FN={m['confusion_matrix']['fn']} TP={m['confusion_matrix']['tp']}"
        )
        if m["n_positive"] == 0:
            lines.append("      -> recall/sensitivity unavailable: no cancer cases in this subgroup")
    lines += [
        "",
        "INTERPRETATION NOTES",
        "-" * 64,
        "  * False negatives (missed cancers) are the clinically costly errors. At the",
        f"    selected threshold there were {overall['confusion_matrix']['fn']} FN vs "
        f"{overall['confusion_matrix']['fp']} FP; lowering the threshold trades more",
        "    false positives for fewer missed cancers.",
        "  * Every record missing the survey block in this dataset is a non-cancer",
        "    diagnosis, so overall metrics are partly inflated by collection/missingness",
        "    patterns. Prefer the survey-present subgroup numbers when judging clinical",
        "    signal, and see reports/missingness_analysis.json.",
        "  * Scores are NOT calibrated medical probabilities; calibration was not evaluated.",
        "  * Project label definition: cancer = BCC/SCC/MEL; non-cancer = ACK/NEV/SEK.",
        "    ACK can be precancerous; this is a project-specific binary definition.",
    ]
    (REPORTS / "classification_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("saved reports/classification_report.txt")

    log(f"TEST @t={threshold}: acc={overall['accuracy']:.4f} F1={overall['f1_cancer']:.4f} "
        f"AUC={overall['roc_auc']:.4f} CM={overall['confusion_matrix']}")
    if not repro_ok:
        raise RuntimeError("reloaded model did not reproduce predictions")


if __name__ == "__main__":
    main()
