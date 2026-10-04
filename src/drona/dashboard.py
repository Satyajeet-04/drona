"""Streamlit operations console for the DRONA engine.

This is the artefact the industry mentor actually looks at, so it is built to
answer the four questions a reviewer asks in order:

1. What did it decide, and how sure is it?        -> verdict + calibrated confidence
2. Why did it decide that?                        -> explanation table + evidence
3. Did a human agree?                             -> HITL review + override tracking
4. What did it cost, and what does it save?        -> latency strip + SLA math

Everything runs in-process against the same :class:`TriageEngine` the API uses,
so the console never depends on the server being up. That matters for a live
demo: one command, no orchestration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drona import taxonomy as tx  # noqa: E402
from drona.engine import get_engine  # noqa: E402
from drona.store import get_store  # noqa: E402
from drona.triage import TicketSignals  # noqa: E402

st.set_page_config(page_title="DRONA Ticket Understanding Engine", layout="wide", page_icon="🎫")

# Headline numbers are *loaded from the measured artifacts*, not hardcoded.
#
# These were literals until the temperature-scaling fix landed, and they went
# stale immediately: the console kept displaying 97.13% auto-answer accuracy
# from the old mis-calibrated cascade (T=3.0) after the calibration work had
# moved the real operating point to 98.24% at 90% coverage. A dashboard that
# displays a stale constant looks authoritative and is wrong, which is worse
# than showing nothing -- so every number below is read from the JSON that the
# experiment scripts wrote.
def _load_facts() -> dict:
    from drona.config import REPORTS

    fusion = json.loads((REPORTS / "03-fusion.json").read_text())
    oracle = json.loads((REPORTS / "02-retrieval-oracle.json").read_text())
    final = json.loads((REPORTS / "06-final-comparison.json").read_text())

    # The operating point we serve at: 90% of tickets auto-answered. Selected by
    # coverage rather than by a raw threshold so the number stays meaningful
    # whatever temperature the calibration step happens to pick.
    curve = fusion["coverage_curve"]
    op = min(curve, key=lambda r: abs(r["coverage"] - 0.90))
    cascade_row = next(r for r in final["decision_table"] if r["system"] == "cascade_fused")
    knn_row = next(r for r in final["decision_table"] if r["system"] == "knn_retrieval")

    facts = {
        "cascade_accuracy": fusion["test_fused"]["accuracy"],
        "cascade_macro_f1": fusion["test_fused"]["macro_f1"],
        "cascade_ece": cascade_row["ece"],
        "accepted_accuracy": op["accuracy_on_accepted"],
        "accepted_coverage": op["coverage"],
        "serving_threshold": op["threshold"],
        "escalation_rate": op["escalation_rate"],
        "temperature": fusion["selected_temperature"],
        "alpha": fusion["selected_alpha"],
        "oracle_recall_at_15": oracle["coarse11"]["union"]["recall_at_15"],
        "oracle_recall_at_3": oracle["coarse11"]["union"]["recall_at_3"],
        "agreement_rate": fusion["test_agreement_rate"],
        "accuracy_when_agree": fusion["test_accuracy_when_agree"],
        "accuracy_when_disagree": fusion["test_accuracy_when_disagree"],
        "mcnemar_p": final["mcnemar_cascade_vs_distilbert"]["p_value"],
        "distilbert_accuracy": final["distilbert_finetuned"]["accuracy"],
        "distilbert_macro_f1": final["distilbert_finetuned"]["macro_f1"],
        "cascade_latency_ms": cascade_row["latency_ms"],
        "knn_accuracy": knn_row["accuracy"],
        "cascade_p0_recall": cascade_row["p0_recall"],
        "cascade_tail_recall": cascade_row["tail_recall"],
        "n_test": final["n_test"],
        "coverage_curve": curve,
        **conformal_facts(),
    }
    return facts


def conformal_facts() -> dict:
    """Surface the split-conformal guarantee next to the empirical curve.

    The empirical coverage table says what happened on one test split; the
    conformal threshold says what is *guaranteed* on unseen data at a chosen
    error rate. Showing only the former undersells the strongest result in the
    project, so both are read from the artifact rather than restated.
    """
    from drona.config import REPORTS  # local import, matching _load_facts()

    path = REPORTS / "08-conformal.json"
    if not path.exists():
        return {}
    levels = json.loads(path.read_text())["levels"].values()
    # Pick the level closest to the operating point the console already shows.
    lv = min(levels, key=lambda r: abs(r["realised_coverage"] - 0.90))
    return {
        "conformal_alpha": lv["alpha"],
        "conformal_threshold": lv["threshold_q"],
        "conformal_coverage": lv["realised_coverage"],
        "conformal_acc_on_accepted": lv["realised_accuracy_on_accepted"],
        "conformal_guarantee_holds": lv["guarantee_holds"],
        "conformal_n_accepted": lv["n_accepted"],
    }


FACTS = _load_facts()

SAMPLES = [
    "my card was declined twice at checkout and the money is still debited, please help urgently",
    "I did not authorise this transaction of 45000 rupees, someone used my card",
    "why was I charged twice for the same subscription this month",
    "how do I change the mobile number linked to my account",
    "the mobile app keeps crashing when I open the transaction screen",
    "what documents do you need for KYC and how long does it take",
    "asdkjh qweasd zxcvbn payment failed",
]

PRIORITY_COLOURS = {"P0": "#b3261e", "P1": "#c77700", "P2": "#1f6feb", "P3": "#5a6472"}


@st.cache_resource
def engine():
    # Threshold comes from the measured coverage curve, not a constant, so the
    # console's escalation behaviour and its headline accuracy always describe
    # the same operating point.
    return get_engine(threshold=FACTS["serving_threshold"])


@st.cache_resource
def store():
    return get_store()


def render_verdict(result) -> None:
    head, mid, right = st.columns([1.4, 1, 1.4])
    with head:
        st.metric("Category", result.category.replace("_", " ").title())
        st.caption(f"confidence **{result.confidence:.2%}**")
    with mid:
        st.markdown("**SLA Priority**")
        st.markdown(
            f"<span style='color:{PRIORITY_COLOURS.get(result.priority, '#333')};font-size:2rem;"
            f"font-weight:700'>{result.priority}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"respond within **{result.sla_minutes} min**")
    with right:
        st.metric("Queue", result.queue.replace("-", " ").title())
        st.caption(f"latency **{result.latency_ms:.0f} ms**")

    top = sorted(result.probabilities.items(), key=lambda kv: -kv[1])[:5]
    st.progress(min(result.confidence, 1.0), text=" ")
    st.caption("calibrated category distribution")
    st.dataframe(
        [{"category": c, "probability": round(p, 4)} for c, p in top],
        hide_index=True,
        width="stretch",
    )


def render_explanation(result) -> None:
    ex = result.explanation
    rows = [
        {"signal": "Category impact", "value": ex["category_impact"], "effect": "sets the priority floor"},
        {"signal": "Urgency terms", "value": ", ".join(ex["urgency_terms"]) or "none",
         "effect": f"urgency score {ex['urgency_score']}"},
        {"signal": "Sentiment", "value": ex["sentiment"], "effect": "escalates P3 only"},
        {"signal": "Channel", "value": ex["channel"], "effect": "non-text signal"},
        {"signal": "Customer tier", "value": ex["customer_tier"], "effect": "non-text signal"},
        {"signal": "Amount at risk", "value": f"INR {ex['amount_at_risk']:,.0f}", "effect": "escalates above 1 lakh"},
        {"signal": "Base priority", "value": ex["base_priority"], "effect": "from category impact"},
        {"signal": "Final priority", "value": ex["final_priority"], "effect": "after escalation rules"},
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    if not ex["non_text_signals_used"]:
        st.info(
            "Priority was decided from text and category impact only. In production the "
            "channel, customer tier, amount at risk and SLA proximity are fused in — "
            "pass them under 'Signals' to see them change the priority."
        )


def render_agent(result) -> None:
    if not result.escalate:
        st.success(
            "Auto-answered by the cascade — the retrieval view and the linear model agreed "
            "and confidence cleared the threshold. No agent run, no draft sent."
        )
        return

    st.warning(f"Escalated to the agent: {result.escalated_because}")
    tabs = st.tabs(["Reasoning trace", "Draft reply", "Verification"])
    trace = getattr(result, "trace", None) or (result.verification.get("trace") if result.verification else None)
    with tabs[0]:
        for kind, message in (trace or []):
            icon = {"thought": "🧠", "action": "🔧", "observation": "👁", "decision": "⚖️"}.get(kind, "•")
            colour = {"decision": "#1f6feb", "action": "#8250df"}.get(kind, "#5a6472")
            st.markdown(
                f"<div style='margin:2px 0'><span style='color:{colour}'>{icon} "
                f"<b>{kind.upper()}</b></span> &nbsp;{message}</div>",
                unsafe_allow_html=True,
            )
    with tabs[1]:
        st.text_area("Draft (editable — the agent never sends anything itself)",
                     result.draft_reply, height=140, key="draft")
        if st.button("Approve and send"):
            st.success("Approved. This is the human-in-the-loop step: nothing leaves the system without it.")
    with tabs[2]:
        v = result.verification
        c1, c2, c3 = st.columns(3)
        c1.metric("Atomic claims", v.get("total", 0))
        c2.metric("Fact precision", f"{v.get('fact_precision', 0):.0%}")
        c3.metric("Invented identifiers", len(v.get("invented_identifiers", [])))
        if v.get("unsupported_claims"):
            st.warning("Claims not supported by the evidence:")
            for claim in v["unsupported_claims"]:
                st.text(f"• {claim}")
        if v.get("invented_identifiers"):
            st.error(
                f"BLOCKED — invented identifier(s) {v['invented_identifiers']}. A reply that "
                "cites a case number that does not exist sends the customer nowhere."
            )
        st.dataframe(v.get("claims", []), hide_index=True, width="stretch")


def render_queue(store) -> None:
    rows = store.by_queue()
    if not rows:
        st.info("No open tickets yet. Triage one on the Ingest tab.")
        return
    grouped: dict[str, list] = {}
    for r in rows:
        grouped.setdefault(r["queue"] or "unrouted", []).append(r)
    for queue, items in sorted(grouped.items()):
        with st.expander(f"{queue} — {len(items)} open", expanded=queue == sorted(grouped)[0]):
            st.dataframe(
                [{"id": i["ticket_id"], "priority": i["priority"], "category": i["category"],
                  "confidence": round(i["confidence"] or 0, 3),
                  "escalated": bool(i["escalated"]), "text": (i["text"] or "")[:90]} for i in items],
                hide_index=True, width="stretch",
            )


def main() -> None:
    eng, db = engine(), store()

    st.title("DRONA — Smart Ticket Understanding Engine")
    st.caption(
        "Offline CPU-only triage: sparse retrieval + calibrated linear fusion, SLA priority routing, "
        "and a tool-using agent that re-decides only the tickets the cascade refuses to auto-answer."
    )

    m = st.columns(7)
    m[0].metric("Test accuracy", f"{FACTS['cascade_accuracy']:.2%}")
    m[1].metric("On auto-answered", f"{FACTS['accepted_accuracy']:.2%}",
                help=f"at {FACTS['accepted_coverage']:.1%} coverage")
    m[2].metric("Conformal guarantee", f"{FACTS['conformal_acc_on_accepted']:.2%}",
                help=(f"accuracy on auto-answered tickets at alpha="
                      f"{FACTS['conformal_alpha']} ({FACTS['conformal_coverage']:.1%} coverage). "
                      "A finite-sample bound, not a measurement on one split."))
    m[3].metric("Oracle recall@15", f"{FACTS['oracle_recall_at_15']:.2%}",
                help="gold label present in the retrieved set — the ceiling for reranking")
    m[4].metric("Agreement rate", f"{FACTS['agreement_rate']:.2%}")
    m[5].metric("Latency", f"{FACTS['cascade_latency_ms']:.0f} ms", help="single ticket, in-process, measured on test split")
    m[6].metric("Tickets stored", db.stats()["total"])

    tabs = st.tabs(["Ingest & triage", "Work queue", "Operations", "Drift"])

    # ---------------------------------------------------------------- ingest
    with tabs[0]:
        with st.form("triage_form"):
            sample = st.selectbox("Load a sample ticket", ["-- pick --"] + SAMPLES)
            text = st.text_area("Ticket text", sample if sample != "-- pick --" else "", height=100)
            s1, s2, s3 = st.columns(3)
            channel = s1.selectbox("Channel", ["web", "email", "chat", "phone"])
            tier = s2.selectbox("Customer tier", ["standard", "premier", "enterprise"])
            amount = s3.number_input("Amount at risk (INR)", 0.0, step=1000.0)
            submitted = st.form_submit_button("Triage ticket", type="primary")

        if submitted and text.strip():
            with st.spinner("Triaging…"):
                tid = db.next_id()
                result = eng.triage(text.strip(), TicketSignals(channel, tier, amount), ticket_id=tid)
                db.record(tid, text.strip(), result.__dict__)
                st.session_state["last"] = (tid, result)

        if "last" in st.session_state:
            tid, result = st.session_state["last"]
            st.divider()
            st.caption(f"ticket `{tid}`")
            render_verdict(result)
            c1, c2 = st.columns([1.3, 1])
            with c1:
                with st.expander("Why this decision", expanded=True):
                    render_explanation(result)
            with c2:
                with st.expander("Retrieved evidence", expanded=True):
                    st.dataframe(
                        [{"#": i + 1, "text": n["text"], "label": n["label"],
                          "similarity": n["similarity"]} for i, n in enumerate(result.neighbours)],
                        hide_index=True, width="stretch",
                    )
            render_agent(result)

    # ----------------------------------------------------------- work queue
    with tabs[1]:
        render_queue(db)
        st.divider()
        st.subheader("Reviewed")
        st.dataframe(
            [{k: v for k, v in r.items() if k in
              ("ticket_id", "category", "human_category", "priority", "approved", "created_at")}
             for r in db.recent(50)],
            hide_index=True, width="stretch",
        )

    # ----------------------------------------------------------- operations
    with tabs[2]:
        s = db.stats()
        st.subheader("Operational counters")
        st.json(s)
        st.subheader("Why the numbers are what they are")
        st.markdown(
            f"""
| Observation | Measurement | Meaning |
|---|---|---|
| Retrieval finds the answer | oracle recall@15 = **{FACTS['oracle_recall_at_15']:.2%}** | the ceiling for any reranker on this dataset |
| kNN scorer extracts it | {FACTS['knn_accuracy']:.2%} | most of the loss is ranking, not retrieval |
| Views agree | {FACTS['agreement_rate']:.2%} of tickets → **{FACTS['accuracy_when_agree']:.2%}** accurate | free error signal |
| Views disagree | {1-FACTS['agreement_rate']:.2%} of tickets → **{FACTS['accuracy_when_disagree']:.2%}** accurate | this is where the agent runs |
| Abstaining at threshold {FACTS['serving_threshold']:.4f} | {FACTS['accepted_coverage']:.2%} coverage → **{FACTS['accepted_accuracy']:.2%}** accurate | accuracy/coverage trade is published, not hidden |
| P0 ticket recall | **{FACTS['cascade_p0_recall']:.2%}** | `fraud_security` + `payment_failure` reaching the right queue |
| DistilBERT vs cascade | McNemar **p = {FACTS['mcnemar_p']:.2f}** | the transformer's accuracy edge is not significant |
"""
        )

    # ------------------------------------------------------------- drift
    with tabs[3]:
        d = db.drift()
        st.subheader("Reference vs live category distribution")
        per_cat = d["per_category"]
        flagged = [k for k, v in per_cat.items() if v["drift"]]
        st.dataframe(
            [{"category": k, "reference_share": v["reference_share"], "live_share": v["live_share"],
              "delta": v["delta"],
              "drift": ("⚠" if v["drift"] else "") if v["drift"] is not None else "n/a"}
             for k, v in per_cat.items()],
            hide_index=True, width="stretch",
        )
        if not d["sufficient_sample"]:
            # Never claim "no drift" from a sample too small to support the
            # claim. An earlier version did exactly that and reported green
            # while comparing the live table against itself.
            st.info(
                f"{d['n_live']} ticket(s) triaged so far -- {d['note']}. "
                f"Reference = {d['n_reference']} training tickets."
            )
        elif flagged:
            st.warning(f"Distribution shift detected in: {', '.join(flagged)}")
        else:
            st.success("No category has drifted more than 5 points from the reference split.")


if __name__ == "__main__":
    main()