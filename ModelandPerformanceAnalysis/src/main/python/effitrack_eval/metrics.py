from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    error_rate = _safe_divide(fp + fn, tp + tn + fp + fn)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error_rate": error_rate,
    }


def summarise_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    metrics = ("accuracy", "precision", "recall", "f1", "error_rate")

    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))

    summary["tp_total"] = float(sum(int(row["tp"]) for row in rows))
    summary["tn_total"] = float(sum(int(row["tn"]) for row in rows))
    summary["fp_total"] = float(sum(int(row["fp"]) for row in rows))
    summary["fn_total"] = float(sum(int(row["fn"]) for row in rows))
    return summary
