"""
src/evaluation/metrics.py
-------------------------
Evaluation metrics for Phase A paddy disease detection.

Provides:
    evaluate()              — compute all metrics from labels + predictions
    is_ready_for_phase_b()  — check Phase A accuracy thresholds
    print_classification_report() — formatted console output

Phase A thresholds (from config.yaml → phase_a_thresholds):
    Leaf Blast F1  ≥ 0.90
    Brown Spot F1  ≥ 0.90
    Overall Acc    ≥ 0.92

Usage:
    from src.evaluation.metrics import evaluate, is_ready_for_phase_b
    results = evaluate(all_labels, all_preds, class_names, config)
    ready   = is_ready_for_phase_b(results, config)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    all_labels: List[int],
    all_preds:  List[int],
    class_names: List[str],
    config: Optional[dict] = None,
) -> Dict:
    """
    Compute full set of classification metrics.

    Parameters
    ----------
    all_labels  : ground-truth class indices
    all_preds   : predicted class indices
    class_names : ordered list of class names (index → name)
    config      : config dict (used to read phase_a_thresholds)

    Returns
    -------
    dict with keys:
        accuracy        : float
        precision_macro : float
        recall_macro    : float
        f1_macro        : float
        f1_weighted     : float
        per_class       : {class_name: {precision, recall, f1, support}}
        confusion_matrix: np.ndarray  (num_classes × num_classes)
        class_names     : list[str]
        phase_b_ready   : bool  (True when Phase A thresholds met)
    """
    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    num_classes = len(class_names)

    # ── Aggregate metrics ─────────────────────────────────────────────
    acc        = accuracy_score(labels_arr, preds_arr)
    prec_macro = precision_score(labels_arr, preds_arr, average="macro",  zero_division=0)
    rec_macro  = recall_score(labels_arr,   preds_arr, average="macro",  zero_division=0)
    f1_macro   = f1_score(labels_arr,       preds_arr, average="macro",  zero_division=0)
    f1_w       = f1_score(labels_arr,       preds_arr, average="weighted", zero_division=0)

    # ── Per-class metrics ─────────────────────────────────────────────
    prec_per   = precision_score(labels_arr, preds_arr, average=None, zero_division=0)
    rec_per    = recall_score(labels_arr,    preds_arr, average=None, zero_division=0)
    f1_per     = f1_score(labels_arr,        preds_arr, average=None, zero_division=0)
    support    = np.bincount(labels_arr, minlength=num_classes)

    per_class: Dict[str, Dict] = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision" : float(prec_per[i]) if i < len(prec_per) else 0.0,
            "recall"    : float(rec_per[i])  if i < len(rec_per)  else 0.0,
            "f1"        : float(f1_per[i])   if i < len(f1_per)   else 0.0,
            "support"   : int(support[i]),
        }

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(labels_arr, preds_arr, labels=list(range(num_classes)))

    # ── Phase B readiness ─────────────────────────────────────────────
    results = {
        "accuracy"        : float(acc),
        "precision_macro" : float(prec_macro),
        "recall_macro"    : float(rec_macro),
        "f1_macro"        : float(f1_macro),
        "f1_weighted"     : float(f1_w),
        "per_class"       : per_class,
        "confusion_matrix": cm,
        "class_names"     : class_names,
    }
    results["phase_b_ready"] = is_ready_for_phase_b(results, config)

    logger.info(
        "Evaluation — acc=%.4f | f1_macro=%.4f | phase_b_ready=%s",
        acc, f1_macro, results["phase_b_ready"],
    )
    return results


# ---------------------------------------------------------------------------
# Phase B gate
# ---------------------------------------------------------------------------

def is_ready_for_phase_b(results: Dict, config: Optional[dict] = None) -> bool:
    """
    Check whether Phase A accuracy thresholds are met.

    Thresholds (read from config.yaml → phase_a_thresholds):
        leaf_blast_f1   ≥ 0.90
        brown_spot_f1   ≥ 0.90
        overall_accuracy ≥ 0.92

    Parameters
    ----------
    results : dict returned by evaluate()
    config  : config dict (uses defaults if None)

    Returns
    -------
    bool — True if all three thresholds are satisfied
    """
    if config:
        thr = config.get("phase_a_thresholds", {})
    else:
        thr = {}

    lb_thr  = thr.get("leaf_blast_f1",    0.90)
    bs_thr  = thr.get("brown_spot_f1",    0.90)
    acc_thr = thr.get("overall_accuracy",  0.92)

    per_class = results.get("per_class", {})
    acc       = results.get("accuracy",  0.0)

    lb_f1  = per_class.get("Leaf Blast",  {}).get("f1", 0.0)
    bs_f1  = per_class.get("Brown Spot",  {}).get("f1", 0.0)

    checks = {
        "Leaf Blast F1"    : (lb_f1,  lb_thr),
        "Brown Spot F1"    : (bs_f1,  bs_thr),
        "Overall Accuracy" : (acc,    acc_thr),
    }

    all_met = True
    for metric, (value, threshold) in checks.items():
        met = value >= threshold
        status = "✅" if met else "❌"
        logger.info("  %s %s: %.4f (threshold ≥ %.2f)", status, metric, value, threshold)
        if not met:
            all_met = False

    return all_met


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_classification_report(results: Dict) -> None:
    """Print a formatted classification report to stdout."""
    class_names = results["class_names"]
    per_class   = results["per_class"]

    print("\n" + "=" * 65)
    print("  PHASE A CLASSIFICATION REPORT")
    print("=" * 65)
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 60)
    for name in class_names:
        m = per_class[name]
        print(
            f"  {name:<20} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} "
            f"{m['support']:>10d}"
        )
    print("  " + "-" * 60)
    print(f"  {'Macro avg':<20} "
          f"{results['precision_macro']:>10.4f} "
          f"{results['recall_macro']:>10.4f} "
          f"{results['f1_macro']:>10.4f}")
    print(f"  {'Accuracy':<20} {results['accuracy']:>10.4f}")
    print(f"  {'F1 weighted':<20} {results['f1_weighted']:>10.4f}")
    print("=" * 65)

    # Phase B gate
    ready = results.get("phase_b_ready", False)
    if ready:
        print("  🌟 PHASE B READY — All accuracy thresholds met!")
    else:
        print("  ⏳ Phase B NOT ready — continue training to hit targets.")
    print("=" * 65 + "\n")

