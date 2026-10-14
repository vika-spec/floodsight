"""Segmentation metrics shared by model evaluation and the EMSR927 case-study comparison."""
from __future__ import annotations

import numpy as np


def confusion(pred: np.ndarray, truth: np.ndarray, n: int, ignore: int | None = None) -> np.ndarray:
    p, t = np.asarray(pred).ravel(), np.asarray(truth).ravel()
    m = (t >= 0) & (t < n) if ignore is None else (t != ignore) & (t < n)
    return np.bincount(n * t[m].astype(np.int64) + p[m].astype(np.int64), minlength=n * n).reshape(n, n)


def scores(cm: np.ndarray, names=None) -> dict:
    n = cm.shape[0]
    per = {}
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        per[c] = {
            "name": names[c] if names else str(c),
            "iou": float(tp / max(tp + fp + fn, 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "support_px": int(cm[c, :].sum()),
        }
    present = [c for c in range(n) if cm[c, :].sum() + cm[:, c].sum() > 0]   # skip classes absent in both
    miou = float(np.mean([per[c]["iou"] for c in present])) if present else 0.0
    return {"per_class": per, "miou": miou,
            "accuracy": float(np.trace(cm) / max(cm.sum(), 1))}


def binary_scores(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> dict:
    p, t = pred.astype(bool), truth.astype(bool)
    if valid is not None:
        p, t = p[valid], t[valid]
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum()); tn = int((~p & ~t).sum())
    return {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
