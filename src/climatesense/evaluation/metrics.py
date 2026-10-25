"""Classification metrics for the four-class ordinal risk target."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score)

N_CLASSES = 4


def compute_metrics(y_true, proba, n_classes: int = N_CLASSES) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=np.float64), 1e-7, 1.0)
    proba = proba / proba.sum(1, keepdims=True)
    y_pred = proba.argmax(1)
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = [c for c in labels if (y_true == c).any() and (y_true != c).any()]
    onehot = np.eye(n_classes)[y_true]
    roc = {c: float(roc_auc_score(onehot[:, c], proba[:, c])) for c in present}
    pr = {c: float(average_precision_score(onehot[:, c], proba[:, c])) for c in present}
    brier = float(np.mean(np.sum((proba - onehot) ** 2, 1)))
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        # undefined (None) if a class is absent from the evaluated split
        "roc_auc_macro_ovr": float(np.mean(list(roc.values()))) if len(roc) == n_classes else None,
        "pr_auc_macro": float(np.mean(list(pr.values()))) if len(pr) == n_classes else None,
        "log_loss": float(log_loss(y_true, proba, labels=labels)),
        "brier": brier,
        "quadratic_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mean_abs_ordinal_error": float(np.mean(np.abs(y_true - y_pred))),
        "per_class": {
            str(c): {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]), "support": int(s[c]),
                     "roc_auc": roc.get(c), "pr_auc": pr.get(c)} for c in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    return out


def bootstrap_ci(y_true, proba, metric="macro_f1", n_boot=1000, seed=0, groups=None) -> tuple:
    """Percentile bootstrap CI.  If `groups` (e.g. site ids) is given, resample whole groups
    (block bootstrap) to respect within-site autocorrelation."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    y_pred = proba.argmax(1)
    vals = []
    if groups is not None:
        ug = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in ug}
    for _ in range(n_boot):
        if groups is None:
            ix = rng.integers(0, len(y_true), len(y_true))
        else:
            ix = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
        if metric == "macro_f1":
            vals.append(f1_score(y_true[ix], y_pred[ix], average="macro", labels=list(range(N_CLASSES)), zero_division=0))
        elif metric == "accuracy":
            vals.append(np.mean(y_true[ix] == y_pred[ix]))
        elif metric == "balanced_accuracy":
            vals.append(balanced_accuracy_score(y_true[ix], y_pred[ix]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(y_true, proba_a, proba_b, n_boot=1000, seed=0, groups=None) -> dict:
    """Block-bootstrap distribution of macro-F1(A) - macro-F1(B) on the same samples."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pa, pb = np.asarray(proba_a).argmax(1), np.asarray(proba_b).argmax(1)
    labels = list(range(N_CLASSES))
    ug = np.unique(groups) if groups is not None else None
    members = {g: np.where(groups == g)[0] for g in ug} if groups is not None else None
    diffs = []
    for _ in range(n_boot):
        ix = (np.concatenate([members[g] for g in rng.choice(ug, len(ug))]) if groups is not None
              else rng.integers(0, len(y_true), len(y_true)))
        diffs.append(f1_score(y_true[ix], pa[ix], average="macro", labels=labels, zero_division=0)
                     - f1_score(y_true[ix], pb[ix], average="macro", labels=labels, zero_division=0))
    diffs = np.array(diffs)
    point = (f1_score(y_true, pa, average="macro", labels=labels, zero_division=0)
             - f1_score(y_true, pb, average="macro", labels=labels, zero_division=0))
    return {"diff": float(point), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_diff_le_0": float(np.mean(diffs <= 0))}
