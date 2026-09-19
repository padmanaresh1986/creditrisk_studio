
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .schema import DEFAULT_THRESHOLD

@dataclass
class EvaluationBundle:
    metrics: dict[str, float]
    confusion: dict[str, int]

def evaluate_binary(y_true, probabilities, threshold: float = DEFAULT_THRESHOLD) -> EvaluationBundle:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    metrics = {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, probabilities)),
        "PR-AUC": float(average_precision_score(y_true, probabilities)),
    }
    confusion = {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    return EvaluationBundle(metrics=metrics, confusion=confusion)

def evaluate_with_two_thresholds(y_true, probabilities, optimized_threshold: float):
    rows = []
    for threshold in [0.50, float(optimized_threshold)]:
        bundle = evaluate_binary(y_true, probabilities, threshold)
        rows.append({
            "Threshold": threshold,
            **bundle.metrics,
        })
    return pd.DataFrame(rows)

def make_threshold_curve(y_true, probabilities, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for t in thresholds:
        p = (probabilities >= t).astype(int)
        rows.append({
            "Threshold": float(t),
            "Precision": float(precision_score(y_true, p, zero_division=0)),
            "Recall": float(recall_score(y_true, p, zero_division=0)),
            "F1": float(f1_score(y_true, p, zero_division=0)),
        })
    return pd.DataFrame(rows)

def select_f1_threshold(y_true, probabilities):
    curve = make_threshold_curve(y_true, probabilities)
    best = (
        curve.sort_values(["F1", "Recall", "Precision"], ascending=False)
        .iloc[0]
    )
    return float(best["Threshold"]), curve, best.to_dict()
