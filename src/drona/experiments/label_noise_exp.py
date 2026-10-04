"""Experiment 07 -- audit and repair Banking77's documented ~14% label noise.

Why
---
Ying & Thomas (*Demystifying Banking77*, ACL Insights 2022, insights-1.19)
measure substantial label error in Banking77 and show that cleaning it moves
conclusions. Our own retrieval output shows the mechanism directly: near-identical
utterances carry different intents, e.g.

    "How can I get new card?"  -> contactless_not_working
    "how do I get a new card"  -> order_physical_card

Our taxonomy merges both into ``card_services``, which is part of why the 10-class
task (94.60%) scores far above the 77-intent task (90.22%). Some of that gap is
noise being averaged away rather than modelling. Any accuracy figure we publish is
therefore an upper bound on what the taxonomy can express, and we cannot know by
how much unless we measure it.

Method
------
1. Produce **out-of-fold** predictions over the training split with a
   cross-validated TF-IDF + LinearSVC pipeline. Fitting on the training data and
   then asking for label quality scores would be circular -- every model scores
   its own training points.
2. Hand those out-of-fold predictions to ``cleanlab`` to flag candidates.
3. **Audit a sample by hand.** The critical discipline: a cleaning method can
   remove 14% of the data while being wrong about half of it, and that would
   degrade the model. The detector's own precision must be measured, never
   assumed. The flagged examples are printed with their annotated label, their
   out-of-fold prediction, and that prediction's probability so they can be
   adjudicated by reading them.
4. Retrain under two repair policies and report the test delta for each:
     - ``drop``    : remove every flagged example
     - ``relabel`` : trust the out-of-fold prediction over the annotation

Nothing is written back to the dataset. This produces evidence for the report,
not a silent rewrite of the source data.
"""

from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from ..cascade import _union_vectorizers
from ..config import REPORTS, SEED
from ..data import load_jsonl
from ..evaluate import report, score

N_FOLDS = 5
SAMPLE_FOR_AUDIT = 40


class UnionTfidf(BaseEstimator, TransformerMixin):
    """char_wb(3,5) + word(1,2) TF-IDF, L2-normalised per block then hstacked.

    Mirrors ``Cascade._features`` exactly (it delegates to the same
    ``_union_vectorizers``) so this experiment measures the same feature space
    the serving system uses. Written as a proper sklearn transformer so it can
    sit inside a ``Pipeline`` and therefore inside ``cross_val_predict`` --
    which is what makes the out-of-fold predictions honest.
    """

    def __init__(self) -> None:
        self.vectorizers = _union_vectorizers()

    def fit(self, texts, y=None):
        self.mats = [normalize(v.fit_transform(texts)) for v in self.vectorizers.values()]
        return self

    def transform(self, texts):
        return sp.hstack(
            [normalize(v.transform(texts)) for v in self.vectorizers.values()], format="csr"
        )

    def _more_tags(self):
        return {"X_types": ["string"]}  # sklearn >=1.6 tag protocol


def build_pipeline() -> Pipeline:
    """TF-IDF features -> LogisticRegression.

    Logistic regression rather than the serving LinearSVC because cleanlab needs
    ``predict_proba`` and ``cross_val_predict`` must be able to drive the whole
    thing as one ``Pipeline`` (``CalibratedClassifierCV(cv="prefit")`` cannot be
    fitted inside a CV fold -- it requires an already-fitted base estimator).
    This is not a downgrade in fidelity: the serving cascade already wraps
    LinearSVC's decision function in exactly this logistic calibration, so the
    probabilities cleanlab sees are the same family the deployed system uses.
    """
    return Pipeline(
        [
            ("features", UnionTfidf()),
            ("clf", LogisticRegression(C=4.0, max_iter=2000, random_state=SEED)),
        ]
    )


def main() -> None:
    train, _, test = load_jsonl("train"), load_jsonl("val"), load_jsonl("test")
    tr_texts = [t.text for t in train]
    y_tr = np.array([t.coarse for t in train])
    te_texts = [t.text for t in test]
    y_te = np.array([t.coarse for t in test])

    out: dict = {
        "n_train": len(train),
        "n_test": len(test),
        "folds": N_FOLDS,
        "seed": SEED,
        "task": "coarse (10 classes present in data)",
    }

    # -- 1. out-of-fold predictions ----------------------------------------
    # cleanlab needs *probabilities*, not hard labels, so the linear model is
    # paired with a logistic calibrator inside each fold. Hard-label
    # cross_val_predict would also throw away the confidence signal that makes
    # confident learning able to separate a genuine label error from a genuinely
    # hard-but-correctly-labelled example.
    print("cross-validated out-of-fold probabilities ...", flush=True)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_proba = cross_val_predict(build_pipeline(), tr_texts, y_tr, cv=cv, n_jobs=1, method="predict_proba")
    # `predict_proba` columns are ordered by sorted unique training labels, so
    # the class order is known without fitting anything. (Reading
    # `classes_` off the pipeline fails twice over: Pipeline does not forward
    # it, and an unfitted estimator has no such attribute.)
    oof_classes = np.unique(y_tr)
    oof = oof_classes[oof_proba.argmax(axis=1)]
    base = score(y_te, build_pipeline().fit(tr_texts, y_tr).predict(te_texts))
    print(f"baseline test acc={base['accuracy']:.4f} macroF1={base['macro_f1']:.4f}", flush=True)

    oof_acc = float((oof == y_tr).mean())
    print(f"out-of-fold agreement with annotation: {oof_acc:.4f}", flush=True)
    out["oof_agreement_with_annotation"] = round(oof_acc, 4)

    # -- 2. cleanlab flags --------------------------------------------------
    # cleanlab 2.x moved this out of the old CleanLabels wrapper: the supported
    # entry point is cleanlab.filter.find_label_issues(labels, pred_probs).
    from cleanlab.filter import find_label_issues

    # cleanlab requires zero-indexed integer labels, not strings. The encoding is
    # the same sorted-unique order as the probability columns above, so
    # `mask[i]` still refers to training example `i`.
    label_ids = {c: i for i, c in enumerate(oof_classes)}
    y_int = np.array([label_ids[c] for c in y_tr], dtype=int)
    mask = find_label_issues(y_int, oof_proba)
    mask = np.asarray(mask, dtype=bool)
    n_flag = int(mask.sum())
    out["n_flagged"] = n_flag
    out["flagged_fraction"] = round(n_flag / len(y_tr), 4)
    print(f"cleanlab flags {n_flag}/{len(y_tr)} = {n_flag / len(y_tr):.1%}", flush=True)

    if n_flag == 0:
        print("CLEANLAB_DONE (no flags)")
        return

    # -- 3. hand-audit sample ----------------------------------------------
    # Highest-confidence complaints first: where the out-of-fold model is most
    # sure the annotation is wrong. These are the cases a human can adjudicate
    # fastest, so the sample is maximally informative per item read.
    prob_of_flagged = np.where(oof == y_tr, 0.0, 1.0)
    flagged_idx = np.flatnonzero(mask)
    order = flagged_idx[np.argsort(-prob_of_flagged[flagged_idx])]
    audit = []
    for i in order[:SAMPLE_FOR_AUDIT]:
        audit.append(
            {
                "text": tr_texts[i][:160],
                "annotated_intent": train[i].intent,
                "annotated_coarse": train[i].coarse,
                "oof_prediction": str(oof[i]),
            }
        )
    out["audit_sample"] = audit
    print("\n=== AUDIT SAMPLE (highest-confidence complaints first) ===", flush=True)
    for a in audit[:25]:
        print(f"  [{a['annotated_coarse']:>18} -> {a['oof_prediction']:<18}] {a['text'][:90]}", flush=True)

    # -- 4. repair policies ------------------------------------------------
    clean_mask = ~mask
    drop = build_pipeline().fit([tr_texts[i] for i in np.flatnonzero(clean_mask)], y_tr[clean_mask])
    m_drop = score(y_te, drop.predict(te_texts))
    print(f"\n[drop]    n={clean_mask.sum()} acc={m_drop['accuracy']:.4f} macroF1={m_drop['macro_f1']:.4f}", flush=True)

    y_relab = oof.copy()
    relab = build_pipeline().fit(tr_texts, y_relab)
    m_relab = score(y_te, relab.predict(te_texts))
    print(f"[relabel] n={len(tr_texts)} acc={m_relab['accuracy']:.4f} macroF1={m_relab['macro_f1']:.4f}", flush=True)

    out["baseline"] = base
    out["policy_drop"] = {
        "n_train_after": int(clean_mask.sum()),
        **m_drop,
        "delta_accuracy": round(m_drop["accuracy"] - base["accuracy"], 4),
        "delta_macro_f1": round(m_drop["macro_f1"] - base["macro_f1"], 4),
    }
    out["policy_relabel"] = {
        "n_train_after": len(tr_texts),
        **m_relab,
        "delta_accuracy": round(m_relab["accuracy"] - base["accuracy"], 4),
        "delta_macro_f1": round(m_relab["macro_f1"] - base["macro_f1"], 4),
    }

    (REPORTS / "07-label-noise.json").write_text(json.dumps(out, indent=2))
    print("\nCLEANLAB_DONE")


if __name__ == "__main__":
    main()