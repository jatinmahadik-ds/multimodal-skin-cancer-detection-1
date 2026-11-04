"""
metrics_utils.py
================
Shared metric computation for the clinical-only XGBoost project so that the
training (validation) and evaluation (test) scripts report identical numbers.

Rules:
  * Metrics that are mathematically unavailable (e.g. recall when a subgroup
    contains no positive cases) are reported as None ("unavailable"), never
    fabricated as 0.
  * A trivial majority-class baseline is always computed alongside.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Full binary metric bundle for cancer class = 1."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    majority = 1 if n_pos >= n_neg else 0
    baseline_acc = max(n_pos, n_neg) / max(len(y_true), 1)

    def or_none(v):
        return None if v is None else float(v)

    # Precision: undefined if nothing was predicted positive.
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)) if (tp + fp) > 0 else None
    # Recall: undefined if there are no actual positives.
    recall = float(tp / (tp + fn)) if n_pos > 0 else None
    # Specificity: undefined if there are no actual negatives.
    specificity = float(tn / (tn + fp)) if n_neg > 0 else None
    # F1 for cancer class: requires actual positives to be meaningful.
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)) if n_pos > 0 else None
    # ROC-AUC: needs both classes present.
    roc = float(roc_auc_score(y_true, y_score)) if (n_pos > 0 and n_neg > 0) else None

    macro_f1 = None
    if n_pos > 0 and n_neg > 0:
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "threshold": float(threshold),
        "n_records": int(len(y_true)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_cancer": precision,
        "recall_cancer_sensitivity": recall,
        "specificity": specificity,
        "f1_cancer": f1,
        "macro_f1": macro_f1,
        "roc_auc": roc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "false_negatives": fn,
        "false_positives": fp,
        "majority_class_baseline": {
            "majority_label": int(majority),
            "accuracy_if_always_majority": float(baseline_acc),
            "note": "trivial baseline: always predict the majority class of this subset",
        },
        "unavailable_metrics": [
            name
            for name, v in [
                ("precision_cancer", precision),
                ("recall_cancer_sensitivity", recall),
                ("specificity", specificity),
                ("f1_cancer", f1),
                ("macro_f1", macro_f1),
                ("roc_auc", roc),
            ]
            if v is None
        ],
    }


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    lo: float = 0.05,
    hi: float = 0.95,
    step: float = 0.01,
) -> Dict[str, Any]:
    """
    Choose a decision threshold on VALIDATION data only.

    Maximises cancer-class F1; ties are broken by higher recall, then by
    proximity to the default 0.5 threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    candidates = np.round(np.arange(lo, hi + 1e-9, step), 2)
    rows = []
    for t in candidates:
        m = compute_binary_metrics(y_true, y_score, float(t))
        if m["f1_cancer"] is None:
            continue
        rows.append({
            "threshold": float(t),
            "f1_cancer": m["f1_cancer"],
            "recall": m["recall_cancer_sensitivity"],
            "precision": m["precision_cancer"],
        })
    if not rows:
        return {"threshold": 0.5, "reason": "no candidate threshold yielded a defined F1; falling back to 0.5"}
    best_f1 = max(r["f1_cancer"] for r in rows)
    pool = [r for r in rows if abs(r["f1_cancer"] - best_f1) < 1e-12]
    best_recall = max(r["recall"] for r in pool)
    pool = [r for r in pool if abs(r["recall"] - best_recall) < 1e-12]
    pool.sort(key=lambda r: abs(r["threshold"] - 0.5))
    best = pool[0]
    return {
        "threshold": best["threshold"],
        "validation_f1_cancer": best["f1_cancer"],
        "validation_recall": best["recall"],
        "validation_precision": best["precision"],
        "method": "maximise validation cancer F1; ties -> higher recall, then closest to 0.5",
        "n_candidates": len(rows),
    }


def subgroup_mask(df, column: str, value: bool) -> np.ndarray:
    return (df[column].astype(bool) == value).values
