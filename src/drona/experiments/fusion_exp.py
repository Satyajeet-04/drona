"""Experiment 03 -- the fusion + calibration ablation.

Three questions, one script:

1. Does fusing the linear model with the retrieval vote beat either alone?
2. What is the best mixing weight, and is it the same on validation and test?
3. If we only auto-answer tickets the cascade is confident about, how much
   accuracy do we buy, and what escalation rate does it cost?

All tuning happens on the validation split. The test split is touched exactly
once, at the end, to report.
"""

from __future__ import annotations

import json

import numpy as np

from ..calibration import ece as _ece
from ..calibration import fit_temperature
from ..cascade import Cascade, CascadeConfig, coverage_curve
from ..config import MODELS, REPORTS
from ..data import load_jsonl
from ..evaluate import report, score


def main() -> None:
    train, val, test = load_jsonl("train"), load_jsonl("val"), load_jsonl("test")
    tr_txt = [t.text for t in train]
    tr_lab = [t.coarse for t in train]
    va_txt = [t.text for t in val]
    va_lab = np.array([t.coarse for t in val])
    te_txt = [t.text for t in test]
    te_lab = np.array([t.coarse for t in test])

    out: dict = {"task": "coarse11", "n_train": len(train), "n_val": len(val), "n_test": len(test)}

    # ---- tune alpha and temperature on validation only -------------------
    grid = []
    for alpha in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        cfg = CascadeConfig(alpha=alpha)
        cascade = Cascade(cfg).fit(tr_txt, tr_lab, va_txt, va_lab.tolist())
        p, _, _ = cascade.fused_probs(va_txt)
        m = score(va_lab, cascade.classes_[p.argmax(axis=1)])
        grid.append({"alpha": alpha, "val": m})
        print(f"alpha={alpha:<5} val acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}", flush=True)

    best_alpha = max(grid, key=lambda g: g["val"]["macro_f1"])["alpha"]
    print(f"\nselected alpha={best_alpha} on validation\n", flush=True)

    # ---- temperature sweep on validation, fitted by NLL ------------------
    # NLL is the proper scoring rule for probabilities and is what temperature
    # scaling is for. An earlier version minimised mean(|conf - correct|),
    # which is not ECE and rewards flattening -- see calibration.py.
    cascade = Cascade(CascadeConfig(alpha=best_alpha)).fit(tr_txt, tr_lab, va_txt, va_lab.tolist())
    vp, _, _ = cascade.fused_probs(va_txt)
    cls_pos = {c: i for i, c in enumerate(cascade.classes_.tolist())}
    va_idx = np.array([cls_pos[l] for l in va_lab])
    best_temp, temps = fit_temperature(vp, va_idx)
    for row in temps:
        print(
            f"temp={row['temperature']:<5} val_nll={row['val_nll']:.5f} "
            f"ece={row['val_ece']:.4f} mean_conf={row['mean_confidence']:.4f}",
            flush=True,
        )
    print(f"\nselected temperature={best_temp} (min NLL on validation)\n", flush=True)
    cascade.cfg.temperature = best_temp

    # ---- single touch of the test split ---------------------------------
    tp, pl, pk = cascade.fused_probs(te_txt)
    pred = cascade.classes_[tp.argmax(axis=1)]
    names = cascade.classes_.tolist()

    linear_only = score(te_lab, cascade.classes_[pl.argmax(axis=1)])
    knn_only = score(te_lab, cascade.classes_[pk.argmax(axis=1)])
    fused = score(te_lab, pred)
    agree = pl.argmax(axis=1) == pk.argmax(axis=1)

    print("--- TEST (coarse11) ---")
    for name, m in (
        ("linear only", linear_only),
        ("knn only", knn_only),
        ("fused", fused),
    ):
        print(f"{name:14} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
    print(f"linear/knn agreement on test: {agree.mean():.4f}")
    print(f"accuracy when they agree   : {(pred[agree] == te_lab[agree]).mean():.4f}")
    print(f"accuracy when they disagree: {(pred[~agree] == te_lab[~agree]).mean():.4f}")

    report(te_lab, pred, names, "Cascade fusion, coarse11, test")

    # Thresholds are placed at quantiles of the observed confidence
    # distribution rather than on a fixed grid. After calibration the mean
    # confidence is ~0.93, so a fixed 0.30-0.95 grid would collapse to a single
    # operating point. Quantile placement makes the coverage axis the primary
    # quantity and the confidence axis a consequence of it, which is how an
    # operations team actually chooses an operating point ("escalate 8% of
    # tickets") rather than by picking a raw score.
    conf = tp.max(axis=1)
    levels = np.array([1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.87, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50])
    thresholds = np.unique(np.quantile(conf, 1.0 - levels))
    curve = coverage_curve(te_lab, tp, cascade.classes_, thresholds)
    print("\ncoverage curve (target cov / threshold / coverage / escalation / acc-on-accepted / projected):")
    for row in curve:
        if row["accuracy_on_accepted"]:
            print(
                f"  t={row['threshold']:<6} cov={row['coverage']:.3f} "
                f"esc={row['escalation_rate']:.3f} acc={row['accuracy_on_accepted']:.4f} "
                f"e2e={row['projected_end_to_end']:.4f}"
            )
    # Self-check: accuracy-on-accepted must be >= overall accuracy at every
    # threshold, because abstaining only ever removes tickets from the hard
    # tail. A curve that dips below the overall number would mean the ordering
    # by confidence is broken.
    #
    # The tolerance is required, not defensive padding: at full coverage the
    # accepted set is every ticket, so the two numbers are mathematically
    # identical. They differ only because `coverage_curve` rounds to 4 dp while
    # `score()` stays full precision -- 1857/1963 comes out of sklearn as
    # 0.9460010188487009 and rounds to 0.946, a gap of 1.02e-6. Rounding to 4 dp
    # can introduce at most 5e-5 of error, so 5e-5 is the principled bound
    # (1e-6 is NOT enough and silently fails here). Without this the assert
    # kills the run before it persists the calibrated model.
    tol = 5e-5
    bad = [r for r in curve
           if r["accuracy_on_accepted"] is not None
           and r["accuracy_on_accepted"] < fused["accuracy"] - tol]
    assert not bad, f"coverage curve dips below overall accuracy at {bad}"
    hi = max((r for r in curve if r["accuracy_on_accepted"] is not None), key=lambda r: r["accuracy_on_accepted"])
    print(f"\nself-check OK: accuracy on accepted rises from {fused['accuracy']:.4f} (overall, 100% coverage) "
          f"to {hi['accuracy_on_accepted']:.4f} at {hi['coverage']:.1%} coverage "
          f"(threshold {hi['threshold']:.3f})")
    te_idx = np.array([cls_pos[l] for l in te_lab])
    print(f"    test mean confidence {conf.mean():.4f} vs accuracy {fused['accuracy']:.4f} "
          f"-> test ECE {_ece(tp, te_idx):.4f}")

    out.update(
        {
            "alpha_grid_val": grid,
            "selected_alpha": best_alpha,
            "temperature_grid_val": temps,
            "selected_temperature": best_temp,
            "test_linear_only": linear_only,
            "test_knn_only": knn_only,
            "test_fused": fused,
            "test_agreement_rate": float(agree.mean()),
            "test_accuracy_when_agree": float((pred[agree] == te_lab[agree]).mean()),
            "test_accuracy_when_disagree": float((pred[~agree] == te_lab[~agree]).mean()),
            "coverage_curve": curve,
        }
    )
    (REPORTS / "03-fusion.json").write_text(json.dumps(out, indent=2))
    cascade.save(MODELS / "cascade.joblib")
    print("\nwrote 03-fusion.json and models/cascade.joblib")


if __name__ == "__main__":
    main()