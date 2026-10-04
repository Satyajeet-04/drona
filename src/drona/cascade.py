"""The confidence-gated cascade: retrieval + linear model + calibrated abstention.

This module is the heart of the project's thesis. Experiment 02 showed that on
the coarse task the gold label is present in the top-15 retrieved neighbours
99.75% of the time, while a distance-weighted kNN vote extracts only 92.56%.
Retrieval is essentially solved; *ranking the retrieved candidates* is not.
So instead of paying for a bigger encoder, we spend a few milliseconds fusing
three cheap, independent views of the evidence and then abstain on the tickets
where the evidence genuinely conflicts.

Three components
----------------
1. ``LinearSVC`` over the same char+word union TF-IDF space. Linear SVC has no
   ``predict_proba``; its decision function is squashed with a logistic
   calibrator fitted on the validation split.
2. ``KnnRetriever.vote`` -- non-parametric evidence from the labelled training
   tickets.
3. Temperature-scaled fusion of the two probability vectors, with the mixing
   weight and the temperature chosen on validation data only.

The output of :meth:`Cascade.predict` carries a calibrated confidence, which
:meth:`Cascade.accept_mask` turns into an accept/reject decision. Rejected
tickets are the ones an SLM agent should look at -- we report the accuracy /
coverage curve so the escalation rate is a measured quantity rather than a
guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from .evaluate import score
from .retrieval import KnnRetriever


def _union_vectorizers() -> dict[str, TfidfVectorizer]:
    return {
        "char": TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
        ),
        "word": TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        ),
    }


@dataclass
class CascadeConfig:
    """Hyperparameters. Every field is tuned on the validation split."""

    alpha: float = 0.5  # weight on the linear model vs the kNN vote
    temperature: float = 1.0  # post-hoc temperature on the fused distribution
    k: int = 15  # neighbours retrieved per ticket
    knn_power: float = 3.0  # distance-weight exponent for the kNN vote
    c: float = 1.0  # LinearSVC regularisation


@dataclass
class Prediction:
    """A single triaged ticket. Everything the dashboard needs."""

    text: str
    label: str
    confidence: float
    p_linear: str
    p_knn: str
    agree: bool
    abstain: bool
    priority: str = "P3"
    queue: str = "general_enquiry"
    top_neighbours: list[dict] = field(default_factory=list)


class Cascade:
    """Fitted retrieval + linear + calibrated fusion cascade."""

    def __init__(self, config: CascadeConfig | None = None) -> None:
        self.cfg = config or CascadeConfig()
        self.vectorizers = _union_vectorizers()
        self.mats: list = []
        self.clf: CalibratedClassifierCV | None = None
        self.retriever: KnnRetriever | None = None
        self.classes_: np.ndarray | None = None
        self.n_features: int = 0

    # -- feature plumbing ---------------------------------------------------
    def _fit_features(self, texts: list[str]) -> None:
        self.mats = [normalize(v.fit_transform(texts)) for v in self.vectorizers.values()]
        self.n_features = sum(m.shape[1] for m in self.mats)

    def _features(self, texts: list[str]):
        import scipy.sparse as sp

        return normalize(
            sp.hstack([normalize(v.transform(texts)) for v in self.vectorizers.values()], format="csr")
        )

    # -- fitting -----------------------------------------------------------
    def fit(self, train_texts: list[str], train_labels: list[str], val_texts: list[str], val_labels: list[str]):
        """Fit features, the linear model and the calibrator on validation data."""
        y = np.asarray(train_labels)
        self.classes_ = np.unique(y)
        self._fit_features(train_texts)
        base = LinearSVC(C=self.cfg.c, dual=True, max_iter=20000, random_state=0)
        # CalibratedClassifierCV with cv="prefit" is deprecated in modern
        # sklearn; fitting the SVC then calibrating its decision function with a
        # logistic regression on the validation split is the explicit,
        # reproducible equivalent and is what we do here.
        base.fit(self._features(train_texts), y)
        self._svc = base
        decision = base.decision_function(self._features(val_texts))
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        self.calibrator = LogisticRegression(max_iter=10000)
        self.calibrator.fit(decision, np.asarray(val_labels))
        self.retriever = KnnRetriever(train_texts, y)
        return self

    def _linear_probs(self, texts: list[str]) -> np.ndarray:
        d = self._svc.decision_function(self._features(texts))
        if d.ndim == 1:
            d = np.column_stack([-d, d])
        p = self.calibrator.predict_proba(d)
        return self._reorder(p, self.calibrator.classes_)

    def _knn_probs(self, texts: list[str]) -> np.ndarray:
        p, _ = self.retriever.vote(texts, k=self.cfg.k, view="union")
        return self._reorder(p, np.unique(self.retriever.labels))

    def _reorder(self, probs: np.ndarray, source_classes) -> np.ndarray:
        """Map a probability matrix onto ``self.classes_`` ordering.

        ``probs`` columns follow ``source_classes``; the returned matrix follows
        ``self.classes_``. The two orderings differ (the calibration set may not
        contain every class), so alignment has to be explicit.
        """
        target = {c: i for i, c in enumerate(self.classes_.tolist())}
        out = np.zeros((probs.shape[0], len(target)))
        for j, c in enumerate(np.asarray(source_classes).tolist()):
            if c in target:
                out[:, target[c]] = probs[:, j]
        return out

    # -- inference ---------------------------------------------------------
    def fused_probs(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pl, pk = self._linear_probs(texts), self._knn_probs(texts)
        fused = self.cfg.alpha * pl + (1 - self.cfg.alpha) * pk
        return self._temperature(fused), pl, pk

    def _temperature(self, probs: np.ndarray) -> np.ndarray:
        t = max(self.cfg.temperature, 1e-3)
        logits = np.log(np.clip(probs, 1e-12, None)) / t
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, texts: list[str], threshold: float = 0.0) -> list[Prediction]:
        fused, pl, pk = self.fused_probs(texts)
        idx = fused.argmax(axis=1)
        out = []
        for i, txt in enumerate(texts):
            li, ki = int(pl[i].argmax()), int(pk[i].argmax())
            conf = float(fused[i, idx[i]])
            out.append(
                Prediction(
                    text=txt,
                    label=str(self.classes_[idx[i]]),
                    confidence=conf,
                    p_linear=str(self.classes_[li]),
                    p_knn=str(self.classes_[ki]),
                    agree=li == ki,
                    abstain=conf < threshold,
                )
            )
        return out

    def accept_mask(self, texts: list[str], threshold: float, require_agreement: bool = True) -> np.ndarray:
        fused, pl, pk = self.fused_probs(texts)
        conf = fused.max(axis=1)
        ok = conf >= threshold
        if require_agreement:
            ok &= pl.argmax(axis=1) == pk.argmax(axis=1)
        return ok

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "cfg": self.cfg,
                "vectorizers": self.vectorizers,
                "svc": self._svc,
                "calibrator": self.calibrator,
                "classes": self.classes_,
            },
            path,
            compress=3,
        )

    @classmethod
    def load(cls, path: Path, retriever: KnnRetriever) -> "Cascade":
        import joblib

        blob = joblib.load(path)
        obj = cls(blob["cfg"])
        obj.vectorizers = blob["vectorizers"]
        obj._svc = blob["svc"]
        obj.calibrator = blob["calibrator"]
        obj.classes_ = blob["classes"]
        obj.mats = [normalize(v.transform([""])) for v in obj.vectorizers.values()]
        # Restore the feature count: `fit()` sets it, `load()` must too, or
        # /health reports n_features=0 and the audit trail understates the model.
        obj.n_features = sum(len(v.vocabulary_) for v in obj.vectorizers.values())
        obj.retriever = retriever
        return obj


def coverage_curve(y_true, probs, classes, thresholds=None, escalated_accuracy: float = 1.0) -> list[dict]:
    """Accuracy as a function of how much of the traffic we are willing to auto-answer.

    This is the central experiment: it shows that abstaining on the hard tail
    buys accuracy at a known, *measured* escalation rate, instead of the opaque
    "confidence threshold" that production vendors use and never publish.

    ``probs`` columns follow ``classes``; ``y_true`` holds the same labels.
    ``escalated_accuracy`` is the assumed accuracy of the SLM-plus-human path on
    the tickets we hand off, used for the projected end-to-end number.
    """
    classes = np.asarray(classes)
    labels = classes[probs.argmax(axis=1)]
    correct = (labels == y_true).astype(float)
    conf = probs.max(axis=1)
    total = len(conf)
    rows = []
    for t in thresholds if thresholds is not None else np.arange(0.30, 1.0, 0.02):
        # Accuracy is computed directly on the accepted mask. Deriving it from a
        # cumulative sum over the confidence-sorted array assumes the accepted
        # set is exactly the top-n, which is false whenever confidences tie.
        mask = conf >= t
        n = int(mask.sum())
        n_correct = float(correct[mask].sum())
        rows.append(
            {
                "threshold": round(float(t), 4),
                "coverage": round(n / total, 4),
                "escalation_rate": round(1 - n / total, 4),
                "accuracy_on_accepted": round(n_correct / n, 4) if n else None,
                # Projected end-to-end accuracy if every escalated ticket is
                # resolved at `escalated_accuracy`.
                "projected_end_to_end": round(
                    (n_correct + (total - n) * escalated_accuracy) / total, 4
                ),
            }
        )
    return rows


__all__ = ["Cascade", "CascadeConfig", "Prediction", "coverage_curve"]