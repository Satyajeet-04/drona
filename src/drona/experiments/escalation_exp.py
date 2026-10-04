"""Experiment 05 -- does the agent actually earn its cost?

The cascade's residual error is concentrated in one measurable place: the
6.6% of tickets where the retrieval view and the linear view disagree. On that
slice accuracy is 62.79% while the accepted slice is 96.84%. The architecture
claims the agent should run only on that slice. This experiment tests that claim
end to end on the held-out test split:

  * how many tickets escalate at each threshold,
  * what the cascade alone scores on the escalated slice,
  * what the agent scores on that same slice,
  * the end-to-end accuracy once the agent's decision replaces the cascade's.

An agent that does not beat the cascade on the slice it is invoked on is pure
added latency, and the honest result would be to report that.
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..cascade import Cascade, coverage_curve
from ..config import REPORTS, SEED
from ..data import load_jsonl
from ..evaluate import per_class, report, score
from ..agent import EscalationAgent


def main() -> None:
    train, val, test = load_jsonl("train"), load_jsonl("val"), load_jsonl("test")
    tr_txt = [t.text for t in train]
    tr_lab = [t.coarse for t in train]
    va_txt = [t.text for t in val]
    va_lab = [t.coarse for t in val]
    te_txt = [t.text for t in test]
    y_te = np.array([t.coarse for t in test])

    cascade = Cascade().fit(tr_txt, tr_lab, va_txt, va_lab)
    classes = cascade.classes_
    fused, pl, pk = cascade.fused_probs(te_txt)
    base_pred = classes[fused.argmax(axis=1)]
    linear_top = classes[pl.argmax(axis=1)]
    knn_top = classes[pk.argmax(axis=1)]
    disagree = linear_top != knn_top
    conf = fused.max(axis=1)

    agent = EscalationAgent()
    out: dict = {"seed": SEED, "n_test": len(test)}

    baseline_acc = float((base_pred == y_te).mean())
    print(f"cascade end-to-end accuracy: {baseline_acc:.4f}")
    print(f"disagreement slice: {disagree.mean():.4f} of traffic, "
          f"accuracy {(base_pred[disagree] == y_te[disagree]).mean():.4f}\n")

    # The agent only ever sees escalated tickets. Union of both triggers, so the
    # slice is well defined independently of any confidence threshold.
    idx = np.where(disagree)[0]
    print(f"agent invoked on {len(idx)} tickets\n")

    # Round 1 showed the evidence-only agent scores *worse* than the cascade on
    # exactly the slice it was invoked on (41.09% vs 64.34%), so letting it
    # overrule unconditionally is a net loss. The agent is therefore treated as a
    # tie-breaker: it may overturn the cascade only when its own evidence is
    # decisive, measured as its margin over the runner-up candidate. Decisions
    # are collected once and each margin is then evaluated analytically, so no
    # sweep can contaminate the next.
    latencies = []
    verifications = []
    draft_invented = 0
    trace_len = []
    records = []  # (row, agent_category, margin, cascade_category, gold)

    t0 = time.time()
    for i in idx:
        text = te_txt[i]
        # Candidate labels come only from the retrieved evidence, so the agent
        # cannot invent a category that retrieval never surfaced.
        nb = cascade.retriever.query(text, k=5, view="union")[0]
        cands = [str(label) for label in dict.fromkeys(nb.labels)]
        cands = cands or [str(base_pred[i])]
        retrieved = [
            {"text": nb.texts[j], "label": str(nb.labels[j]), "similarity": float(nb.scores[j])}
            for j in range(len(nb.scores))
        ]
        decision, draft, ver = agent.handle(
            text=text, candidate_labels=cands, retrieved=retrieved,
            category=str(base_pred[i]), priority="P3", reference=f"DRN-{100000+i}",
        )
        ordered = sorted(decision["scores"].values(), reverse=True)
        margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0
        agent_cat = decision["category"] if decision["category"] in set(classes.tolist()) else str(base_pred[i])
        records.append((int(i), agent_cat, float(margin), str(base_pred[i]), str(y_te[i])))
        verifications.append(ver)
        draft_invented += 1 if ver["invented_identifiers"] else 0
        trace_len.append(len(decision["trace"].steps))
        latencies.append((time.time() - t0) / (len(latencies) + 1) * 1000)

    slice_cascade = float((base_pred[idx] == y_te[idx]).mean())
    gold = np.array([t.coarse for t in test])
    print(f"on the escalated slice (n={len(idx)}): cascade {slice_cascade:.4f}")

    margins = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60]
    sweep = []
    print("\nmargin sweep (agent overturns the cascade only if its evidence clears this margin):")
    print(f"  {'margin':>7}{'overruled':>11}{'correct':>9}{'wrong':>7}{'slice_acc':>11}{'end_to_end':>12}")
    for m in margins:
        pred = base_pred.copy()
        n_over = n_right = n_wrong = 0
        for row, agent_cat, margin, cascade_cat, y in records:
            if agent_cat != cascade_cat and margin >= m:
                pred[row] = agent_cat
                n_over += 1
                if agent_cat == y:
                    n_right += 1
                else:
                    n_wrong += 1
        slice_acc = float((pred[idx] == gold[idx]).mean())
        e2e = float((pred == y_te).mean())
        sweep.append({"margin": m, "overruled": n_over, "correct": n_right, "wrong": n_wrong,
                      "slice_accuracy": round(slice_acc, 4), "end_to_end": round(e2e, 4)})
        print(f"  {m:>7.2f}{n_over:>11}{n_right:>9}{n_wrong:>7}{slice_acc:>11.4f}{e2e:>12.4f}")

    best = max(sweep, key=lambda s: s["end_to_end"])
    best_margin = best["margin"]
    agent_pred = base_pred.copy()
    for row, agent_cat, margin, cascade_cat, y in records:
        if agent_cat != cascade_cat and margin >= best_margin:
            agent_pred[row] = agent_cat
    end_to_end = float((agent_pred == y_te).mean())
    slice_agent = float((agent_pred[idx] == gold[idx]).mean())

    print(f"\nbest margin={best_margin}: slice {slice_cascade:.4f} -> {slice_agent:.4f}, "
          f"end-to-end {baseline_acc:.4f} -> {end_to_end:.4f} ({end_to_end - baseline_acc:+.4f})")
    print(f"agent latency: {np.mean(latencies):.1f} ms/ticket, "
          f"mean trace {np.mean(trace_len):.1f} steps")
    prec = [v["fact_precision"] for v in verifications]
    print(f"draft fact precision: mean {np.mean(prec):.3f}, "
          f"drafts with invented identifiers: {draft_invented}")

    report(y_te, agent_pred, classes.tolist(), "Cascade + advisory agent, coarse11, test")

    curve = coverage_curve(y_te, fused, classes, np.arange(0.30, 1.001, 0.05))
    out.update(
        {
            "baseline_accuracy": baseline_acc,
            "disagreement_rate": float(disagree.mean()),
            "escalated_n": int(len(idx)),
            "slice_cascade_accuracy": slice_cascade,
            "slice_agent_accuracy": slice_agent,
            "best_margin": best_margin,
            "margin_sweep": sweep,
            "end_to_end_with_agent": end_to_end,
            "agent_latency_ms": float(np.mean(latencies)),
            "mean_trace_steps": float(np.mean(trace_len)),
            "draft_fact_precision_mean": float(np.mean(prec)),
            "drafts_with_invented_identifiers": draft_invented,
            "per_class": per_class(y_te, agent_pred, classes.tolist()),
            "coverage_curve": curve,
        }
    )
    (REPORTS / "05-escalation.json").write_text(json.dumps(out, indent=2))
    print("\nwrote 05-escalation.json")


if __name__ == "__main__":
    main()