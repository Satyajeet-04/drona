"""Shared metrics. Reports macro-F1 alongside accuracy because the synopsis
demands "macro F1 close to accuracy" -- with an 18.7:1 class imbalance those two
diverge sharply, and quoting only accuracy would be misleading."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def score(y_true, y_pred, names=None):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _encode(y_true, y_pred, names):
    """Map string labels onto positions in ``names``.

    Every label array in this project is a string array (``Ticket.coarse`` /
    ``Ticket.intent``), so the sklearn helpers below must be fed integer codes.
    Encoding here keeps callers free of boilerplate and guarantees the same
    ordering is used for precision, recall, F1 and the confusion matrix.
    """
    lookup = {n: i for i, n in enumerate(names)}
    enc = lambda y: np.array([lookup[v] for v in y], dtype=np.int64)
    return enc(y_true), enc(y_pred)


def per_class(y_true, y_pred, names):
    names = list(names)
    yt, yp = _encode(y_true, y_pred, names)
    p = precision_score(yt, yp, average=None, zero_division=0, labels=range(len(names)))
    r = recall_score(yt, yp, average=None, zero_division=0, labels=range(len(names)))
    f = f1_score(yt, yp, average=None, zero_division=0, labels=range(len(names)))
    sup = np.bincount(yt, minlength=len(names))
    rows = []
    for i, n in enumerate(names):
        rows.append((n, int(sup[i]), float(p[i]), float(r[i]), float(f[i])))
    rows.sort(key=lambda r: r[4])
    return rows


def report(y_true, y_pred, names, title):
    m = score(y_true, y_pred)
    print(f"\n=== {title} ===")
    print(f"accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  "
          f"macro_P={m['macro_precision']:.4f}  macro_R={m['macro_recall']:.4f}")
    print(f"{'class':<24}{'n':>6}{'P':>8}{'R':>8}{'F1':>8}")
    for n, sup, p, r, f in per_class(y_true, y_pred, names):
        print(f"{n:<24}{sup:>6}{p:>8.3f}{r:>8.3f}{f:>8.3f}")
    return m


def confusion(y_true, y_pred, names):
    names = list(names)
    yt, yp = _encode(y_true, y_pred, names)
    cm = confusion_matrix(yt, yp, labels=range(len(names)))
    return cm, names