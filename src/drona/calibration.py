"""Probability calibration.

Why this module exists
----------------------
The cascade blends two probability sources that are scaled completely
differently: a LinearSVC decision function squashed through a logistic link
(very peaked) and a distance-weighted 15-neighbour vote (very flat). The blend
is therefore *uncalibrated*, and an uncalibrated confidence is worthless --
which matters here because the whole abstention design ("accept if the two
models agree and confidence clears a threshold") reads that number.

A first attempt selected the temperature by

    mean(|confidence - correct|)

which is not expected calibration error. It is dominated by the accuracy term
rather than the confidence term, so it rewards *flattening* the distribution
toward uniform, and it drove the temperature to the edge of the search grid.
The result was a cascade whose confidences sat near 0.5 while its accuracy was
0.946, i.e. ECE 0.417 -- the model was systematically under-confident and its
abstention thresholds were meaningless.

Temperature scaling is a strictly monotone transform of the logits, so it
cannot change which label is predicted -- it only changes how the probability
is expressed. The correct objective for fitting it is therefore negative
log-likelihood, the proper scoring rule for probabilistic forecasts. ECE is
reported alongside as a diagnostic, but it is not what we optimise.
"""

from __future__ import annotations

import numpy as np


def ece(probs: np.ndarray, y_true_idx: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: confidence-weighted |accuracy - confidence|.

    Args:
        probs: (n, c) probability matrix.
        y_true_idx: (n,) integer class indices matching ``probs`` columns.
        bins: number of equal-width confidence bins.
    """
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true_idx).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def nll(probs: np.ndarray, y_true_idx: np.ndarray) -> float:
    """Mean negative log-likelihood of the assigned probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    p = np.clip(probs[np.arange(len(y_true_idx)), y_true_idx], 1e-12, 1.0)
    return float(-np.log(p).mean())


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    """Rescale a probability matrix by temperature on the log-probabilities."""
    t = max(float(t), 1e-3)
    logits = np.log(np.clip(probs, 1e-12, None)) / t
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(
    probs: np.ndarray,
    y_true_idx: np.ndarray,
    grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0),
) -> tuple[float, list[dict]]:
    """Pick the temperature that minimises validation NLL.

    Returns ``(best_t, sweep)`` where ``sweep`` records every candidate so the
    report can show the objective was minimised rather than clipped.
    """
    sweep = []
    for t in grid:
        p = apply_temperature(probs, t)
        sweep.append(
            {
                "temperature": t,
                "val_nll": round(nll(p, y_true_idx), 5),
                "val_ece": round(ece(p, y_true_idx), 5),
                "mean_confidence": round(float(p.max(axis=1).mean()), 5),
            }
        )
    best = min(sweep, key=lambda r: r["val_nll"])["temperature"]
    return best, sweep


__all__ = ["ece", "nll", "apply_temperature", "fit_temperature"]