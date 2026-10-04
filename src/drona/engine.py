"""The DRONA engine: one object that triages a ticket end to end.

Pipeline
--------
    text
      -> sparse retrieval (char+word union TF-IDF kNN, ~4 ms)
      -> LinearSVC + calibrated fusion, temperature-scaled
      -> abstain if the two views disagree or confidence is below threshold
      -> SLA priority engine (category impact + urgency + non-text signals)
      -> router (category, priority) -> queue
      -> [escalated only] agent: rerank retrieved cases with tools, draft reply
      -> atomic-fact verification of the draft
      -> human-in-the-loop hand-off

Everything runs in-process on CPU with no network calls, so the demo behaves
identically with the internet disconnected.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import taxonomy as tx
from .agent import EscalationAgent
from .cascade import Cascade
from .config import MODELS, PROCESSED
from .data import load_jsonl
from .retrieval import KnnRetriever
from .triage import TicketSignals, TriageResult, assign_priority


def _as_signals(signals) -> TicketSignals:
    """Coerce whatever the caller passed into a `TicketSignals`.

    The REST layer hands over `Signals.model_dump()` -- a plain dict. Passing
    that straight to `assign_priority` raises AttributeError on
    `signals.amount_at_risk`, which turned every request carrying a channel or
    customer tier into an HTTP 500. Normalising here means the dataclass is the
    single contract for every caller, and unknown keys are ignored rather than
    crashing the request.
    """
    if signals is None:
        return TicketSignals()
    if isinstance(signals, TicketSignals):
        return signals
    if isinstance(signals, dict):
        fields = {f for f in TicketSignals.__dataclass_fields__}
        return TicketSignals(**{k: v for k, v in signals.items() if k in fields})
    return TicketSignals(**{k: v for k, v in vars(signals).items() if k in fields})


class TriageEngine:
    """Loads the fitted cascade once and serves tickets from it."""

    def __init__(self, threshold: float = 0.7635, k: int = 5) -> None:
        # Operating point from the *calibrated* coverage curve in experiment 03
        # (`03-fusion.json`): threshold 0.7635 accepts 90% of traffic at 98.24%
        # accuracy on the accepted slice. The earlier value of 0.35 was tuned
        # against the mis-calibrated cascade (T=3.0), where confidences averaged
        # 0.53; after temperature fitting to T=0.75 confidences average 0.93, so
        # 0.35 would have accepted essentially everything and the escalation
        # path would never have fired.
        self.threshold = threshold
        self.k = k
        self.cascade: Cascade | None = None
        self.retriever: KnnRetriever | None = None
        self.agent: EscalationAgent | None = None
        self.ready = False

    # -- loading -----------------------------------------------------------
    def load(self, use_llm: bool = False) -> "TriageEngine":
        train = load_jsonl("train")
        texts = [t.text for t in train]
        labels = np.array([t.coarse for t in train])

        self.retriever = KnnRetriever(texts, labels)
        blob = Path(MODELS / "cascade.joblib")
        if blob.exists():
            # Delegate to Cascade.load rather than re-implementing the unpickle
            # here: this block had drifted from Cascade.load, so fixes applied
            # to one (e.g. restoring n_features) silently missed the other.
            self.cascade = Cascade.load(blob, self.retriever)
        else:  # fall back to refitting so the service is never dead
            val = load_jsonl("val")
            self.cascade = Cascade().fit(
                texts, labels.tolist(), [t.text for t in val], [t.coarse for t in val]
            )
        self.agent = EscalationAgent(enable_llm=use_llm)
        self.ready = True
        return self

    # -- inference ---------------------------------------------------------
    def triage(self, text: str, signals: TicketSignals | None = None, ticket_id: str | None = None) -> TriageResult:
        if not self.ready:
            self.load()
        assert self.cascade is not None and self.retriever is not None
        t0 = time.perf_counter()

        fused, pl, pk = self.cascade.fused_probs([text])
        classes = self.cascade.classes_
        order = np.argsort(-fused[0])
        category = str(classes[order[0]])
        confidence = float(fused[0, order[0]])
        probs = {str(classes[i]): float(fused[0, i]) for i in order}

        linear_top = str(classes[pl[0].argmax()])
        knn_top = str(classes[pk[0].argmax()])
        agree = linear_top == knn_top

        nbrs = self.retriever.query(text, k=self.k, view="union")[0]
        neighbours = [
            {
                "text": nbrs.texts[i],
                "label": str(nbrs.labels[i]),
                "similarity": round(float(nbrs.scores[i]), 4),
            }
            for i in range(len(nbrs.scores))
        ]

        # Escalation gate: disagreement between the two independent views is a
        # far stronger error signal than raw confidence, and it is free.
        escalate = (not agree) or confidence < self.threshold
        reason = ""
        if not agree:
            reason = f"retrieval and linear model disagree ({linear_top} vs {knn_top})"
        elif confidence < self.threshold:
            reason = f"calibrated confidence {confidence:.2f} below threshold {self.threshold}"

        priority, explanation = assign_priority(category, text, _as_signals(signals))

        result = TriageResult(
            text=text,
            ticket_id=ticket_id or "",
            category=category,
            confidence=confidence,
            probabilities=probs,
            priority=priority,
            queue=tx.route(category, priority),
            sla_minutes=int(explanation["sla_minutes"]),
            sentiment=str(explanation["sentiment"]),
            escalate=escalate,
            escalated_because=reason,
            explanation=explanation,
            neighbours=neighbours,
        )

        if escalate and self.agent is not None:
            decision, draft, verification = self.agent.handle(
                text=text,
                candidate_labels=[n["label"] for n in neighbours] or [category],
                retrieved=neighbours,
                category=category,
                priority=priority,
                signals=signals,
                reference=ticket_id or "",
            )
            # ADVISORY ONLY -- deliberately does NOT overwrite the cascade's
            # verdict. Experiment 05 measured the agent at 41.09% on the
            # escalated slice against the cascade's 64.34%, and a margin sweep
            # showed every threshold makes end-to-end accuracy worse (94.70%
            # -> 93.17% at margin 0, -> 94.40% even at margin 0.60). An agent
            # that cannot beat the component it was meant to improve has no
            # business overruling it, so its proposal is surfaced to the human
            # reviewer instead. This is the honest version of "human in the
            # loop": the machine proposes, the cascade's measured evidence
            # stands, and a person decides.
            result.agent_proposal = decision
            result.draft_reply = draft
            result.verification = verification
            result.trace = decision["trace"].steps
            result.escalated_because = reason or "escalated to agent"

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def triage_batch(self, texts: list[str], signals: list[TicketSignals] | None = None) -> list[TriageResult]:
        # Loops over the single-ticket path on purpose. A truly vectorised path
        # would batch the sparse matmul in `Cascade.fused_probs`, but the
        # per-ticket evidence list, priority explanation and optional agent
        # trace all differ per row, so batching only the classifier would save
        # little while adding a second code path that needs its own testing.
        # Kept as a loop; correctness over a premature ~10% throughput win.
        signals = signals or [None] * len(texts)
        return [self.triage(t, s) for t, s in zip(texts, signals)]

    def metrics(self) -> dict:
        """Self-reported card for the dashboard's health strip."""
        return {
            "threshold": self.threshold,
            "k": self.k,
            "n_reference_tickets": len(self.retriever.texts) if self.retriever else 0,
            "n_features": getattr(self.cascade, "n_features", 0),
            "llm_enabled": bool(self.agent and self.agent.llm_enabled),
        }


_ENGINE: TriageEngine | None = None


def get_engine(threshold: float = 0.7635, use_llm: bool = False) -> TriageEngine:
    """Process-wide singleton so the FastAPI app and the dashboard share one fit.

    The default 0.7635 is the calibrated operating point at which 90% of
    tickets are auto-answered and the accepted ones are 98.24% correct
    (see ``03-fusion.json``). It replaced a stale 0.35 that had been tuned
    against the *mis-calibrated* cascade (temperature 3.0, mean confidence
    0.53). After temperature fitting the confidences average 0.93, so 0.35
    would have accepted essentially every ticket and the escalation path
    would never have fired -- silently invalidating the whole
    selective-prediction claim.
    """
    global _ENGINE
    if _ENGINE is None or _ENGINE.threshold != threshold:
        _ENGINE = TriageEngine(threshold=threshold).load(use_llm=use_llm)
    return _ENGINE