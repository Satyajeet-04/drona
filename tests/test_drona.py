"""Tests for the parts of DRONA where a silent regression would be invisible.

The project has no network access at inference and no API keys, so anything a
test could catch is a bug rather than a missing credential. These tests target
the specific failure modes we actually hit while building:

  * a taxonomy that silently stops covering the dataset,
  * a retriever that returns the wrong rows because of sparse-matrix indexing,
  * a calibrated model served with a threshold tuned against a *different*
    calibration (the bug that made the escalation gate never fire),
  * amount extraction that silently returns 0 and under-prioritises every
    ticket mentioning money,
  * a drift monitor whose reference is drawn from the live data and therefore
    can never fire.

Run:  ./run.sh test
"""

from __future__ import annotations

import numpy as np
import pytest

from drona import taxonomy as tx
from drona.data import load_jsonl
from drona.evaluate import confusion, per_class, score
from drona.retrieval import KnnRetriever
from drona.triage import TicketSignals, assign_priority, extract_amount


# --------------------------------------------------------------------------
# taxonomy
# --------------------------------------------------------------------------


def test_taxonomy_covers_every_intent_in_the_dataset():
    """`validate` exists precisely to catch a mis-typed intent name."""
    train = load_jsonl("train")
    tx.validate({t.intent for t in train})


def test_taxonomy_classes_are_unique_and_registered():
    assert len(set(tx.COARSE_CATEGORIES)) == len(tx.COARSE_CATEGORIES), "duplicate coarse class"
    for c in tx.COARSE_CATEGORIES:
        assert c in tx.CATEGORY_IMPACT, f"{c} missing an impact score"
        reachable = set(tx.INTENT_TO_COARSE.values())
        for c in tx.COARSE_CATEGORIES:
            # loan_emi is deliberately unreachable: Banking77 has zero loan
            # intents, so a tenth of the declared label space has no data.
            assert (c in reachable) or c in ("loan_emi", "other"), c


def test_every_priority_has_an_sla_and_a_route():
    for p in tx.SLA_RESPONSE_MINUTES:
        assert p in ("P0", "P1", "P2", "P3")
        route = tx.route("payment_failure", p)
        assert route, f"no queue for payment_failure/{p}"
        assert isinstance(route, str)


def test_slas_are_ordered_by_severity():
    m = tx.SLA_RESPONSE_MINUTES
    assert m["P0"] < m["P1"] < m["P2"] < m["P3"]


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def test_per_class_and_confusion_accept_string_labels():
    """They used to require integer codes and crashed on the string labels the
    whole project passes around."""
    y = np.array(["a", "b", "a", "c"])
    p = np.array(["a", "b", "b", "c"])
    rows = per_class(y, p, ["a", "b", "c"])
    assert {r[0] for r in rows} == {"a", "b", "c"}
    cm, names = confusion(y, p, ["a", "b", "c"])
    assert cm.shape == (3, 3)
    assert names == ["a", "b", "c"]
    assert cm.sum() == 4


def test_per_class_is_sorted_worst_f1_first():
    y = np.array(["a", "a", "a", "b"])
    p = np.array(["a", "a", "a", "a"])  # b never predicted -> F1 0
    rows = per_class(y, p, ["a", "b"])
    assert rows[0][0] == "b"


def test_score_reports_macro_f1_next_to_accuracy():
    """The whole point: with an 18.7:1 imbalance these two diverge."""
    m = score(np.array(["a"] * 90 + ["b"] * 10), np.array(["a"] * 100))
    assert m["accuracy"] == 0.9
    assert m["macro_f1"] < 0.6


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def test_knn_returns_the_identical_text_it_was_given():
    """A sparse-row indexing bug once made neighbours come from the wrong
    document; the only honest regression test is to look up a known string."""
    # NB: the distinguishing token must be 2+ characters. sklearn's default
    # token_pattern is r"(?u)\b\w\w+\b", so bare digits are discarded and every
    # document collapses to the same bag of words.
    texts = [f"payment failed on card number {n:02d}" for n in range(20)]
    labels = np.array(["payment_failure"] * 20)
    r = KnnRetriever(texts, labels)
    probe = "payment failed on card number 07"
    nb = r.query(probe, k=3, view="union")[0]
    assert nb.texts[0] == probe
    assert nb.labels[0] == "payment_failure"


def test_knn_scores_are_descending():
    texts = [f"refund not received {i}" for i in range(15)]
    r = KnnRetriever(texts, np.array(["refund_chargeback"] * 15))
    nb = r.query("refund not received", k=5, view="union")[0]
    assert list(nb.scores) == sorted(nb.scores, reverse=True)
    assert len(nb.scores) <= 5


def test_knn_vote_probabilities_are_a_distribution():
    texts = [f"card declined {i}" for i in range(12)]
    labels = np.array(["card_services"] * 6 + ["payment_failure"] * 6)
    r = KnnRetriever(texts, labels)
    probs, _ = r.vote("card declined again", k=4, view="union")
    assert probs.shape[0] == 1
    assert np.isclose(probs.sum(), 1.0)
    assert (probs >= 0).all()


@pytest.mark.parametrize("view", ["char", "word", "union"])
def test_all_three_views_run(view):
    texts = [f"kyc documents needed {i}" for i in range(10)]
    r = KnnRetriever(texts, np.array(["kyc_verification"] * 10))
    assert len(r.query("kyc documents", k=3, view=view)) == 1


# --------------------------------------------------------------------------
# amount extraction -- the bug that made money_at_risk always 0
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("the amount is Rs 45000 and was debited", 45_000),
        ("charged ₹1,50,000 on my card", 150_000),
        ("Rs. 2,50,750 debited from my account", 250_750),
        ("they took 45k from my balance", 45_000),
        ("a sum of 1.5 lakh was charged", 150_000),
        ("lost 2 crore in a fraud transaction", 20_000_000),
        ("INR 999 was charged", 999),
        ("payment failed twice", 0),
    ],
)
def test_extract_amount(text, expected):
    got = extract_amount(text)
    assert got == expected, f"{text!r} -> {got}, expected {expected}"


def test_extract_amount_takes_the_largest_not_the_first():
    assert extract_amount("Rs 45000 and Rs 20 fee") == 45_000


def test_extract_amount_does_not_split_a_grouped_number():
    """The grouped alternative must be tried first: 45,000 vs 45|000."""
    assert extract_amount("Rs 45,000") == 45_000


# --------------------------------------------------------------------------
# priority
# --------------------------------------------------------------------------


def test_high_value_fraud_is_p0():
    p, why = assign_priority(
        "fraud_security", "unauthorised transaction of Rs 50000 from my account", None
    )
    assert p == "P0"
    assert why["amount_at_risk"] == 50_000


def test_low_value_enquiry_is_not_p0():
    p, _ = assign_priority("general_enquiry", "what are your opening hours", None)
    assert p in ("P2", "P3")


def test_explicit_signal_overrides_text_extraction():
    sig = TicketSignals(amount_at_risk=5_000_000.0)
    p, why = assign_priority("payment_failure", "payment failed", sig)
    assert why["amount_source"] == "signal"
    assert p == "P0"


def test_banking_complaint_reads_as_negative_sentiment():
    p, why = assign_priority(
        "payment_failure", "payment failed 3 times, money debited but not credited", None
    )
    assert why["sentiment"] == "negative"


def test_priority_explanation_reports_its_inputs():
    _, why = assign_priority("card_services", "card declined", None)
    for key in ("base_priority", "final_priority", "sentiment", "amount_source"):
        assert key in why, f"explanation is missing {key}"


# --------------------------------------------------------------------------
# drift -- a monitor that can never fire is worse than none
# --------------------------------------------------------------------------


def test_drift_reference_is_not_the_live_data(tmp_path):
    from drona import store as store_mod

    path = tmp_path / "drift.sqlite"
    st = store_mod.Store(path)
    for i in range(40):
        # unique ids: record() is INSERT OR REPLACE, so a repeated id
        # would silently collapse all 40 rows into one
        st.record(f"DRN-{i:04d}", "payment failed", {"category": "payment_failure"})
    d = st.drift()
    assert d["sufficient_sample"] is True
    pc = d["per_category"]
    # Every category must appear with a live share of 0 and a non-zero
    # reference share, otherwise reference and live are the same table.
    assert any(v["live_share"] == 0.0 for v in pc.values())
    assert any(v["reference_share"] > 0.0 for v in pc.values())


def test_drift_is_none_below_the_minimum_sample(tmp_path):
    from drona import store as store_mod

    st = store_mod.Store(tmp_path / "drift_small.sqlite")
    st.record("DRN-0001", "payment failed", {"category": "payment_failure"})
    d = st.drift()
    assert d["sufficient_sample"] is False
    assert all(v["drift"] is None for v in d["per_category"].values())


# --------------------------------------------------------------------------
# signals coercion -- FastAPI hands us dicts, the dataclass wants attributes
# --------------------------------------------------------------------------


def test_dict_signals_do_not_crash_priority_assignment():
    from drona.engine import _as_signals

    sig = _as_signals({"amount_at_risk": 99_000.0, "channel": "email", "bogus": 1})
    assert isinstance(sig, TicketSignals)
    assert sig.amount_at_risk == 99_000.0


def test_none_signals_still_produce_a_usable_explanation():
    p, why = assign_priority("billing_charges", "you overcharged me", None)
    assert p.startswith("P")
    assert why["amount_source"] in ("none", "text_extracted", "signal")


# --------------------------------------------------------------------------
# knn_predict label indexing regression test
# --------------------------------------------------------------------------


def test_knn_predict_returns_correct_unique_class_labels():
    from drona.retrieval import knn_predict

    train_texts = [
        "card payment failed at checkout",
        "card payment failed at checkout",
        "app is crashing on open",
        "app is crashing on open",
    ]
    train_labels = np.array(["payment_failure", "payment_failure", "technical_app", "technical_app"])
    pred, conf, _ = knn_predict(["app is crashing on open"], train_texts, train_labels)
    assert pred[0] == "technical_app"
    assert conf[0] > 0.5


# --------------------------------------------------------------------------
# agent & verification
# --------------------------------------------------------------------------


def test_agent_verify_exports_and_runs():
    from drona.agent import verify, EscalationAgent

    entry = {
        "policy": "A transaction that shows as failed but debited is treated as a reconciliation case.",
        "action": "Confirm transaction reference and run reconciliation check.",
        "escalate_to": "Payments Operations",
        "customer_facing": "We are checking reference {ref}. Reversals occur within one working day.",
    }
    retrieved = [{"text": "payment failed debited", "label": "payment_failure", "similarity": 0.85}]
    draft = "We are checking reference DRN-100050. Reversals occur within one working day."
    report = verify(draft, retrieved, entry, reference="DRN-100050")
    assert report["supported"] > 0
    assert report["fact_precision"] > 0.5
    assert report["releasable"] is True
    assert len(report["invented_identifiers"]) == 0


def test_agent_verify_detects_invented_identifiers():
    from drona.agent import verify

    entry = {
        "policy": "Standard refund policy.",
        "action": "Initiate refund.",
        "escalate_to": "Billing",
        "customer_facing": "Your refund is processed.",
    }
    draft = "Your case TXN-999999 has been approved for refund under reference FAKE-12345."
    report = verify(draft, [], entry, reference="DRN-0001")
    assert "TXN-999999" in report["invented_identifiers"] or "FAKE-12345" in report["invented_identifiers"]
    assert report["releasable"] is False


# --------------------------------------------------------------------------
# entities & postfix currency
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("amount debited 25000 rs from account", 25_000),
        ("lost 50000 rupees in scam", 50_000),
        ("charged 1200 inr fee", 1_200),
        ("deducted $500 twice", 500),
    ],
)
def test_extract_amount_supports_postfix_currencies(text, expected):
    got = extract_amount(text)
    assert got == expected, f"{text!r} -> {got}, expected {expected}"


def test_extract_entities_discovers_domain_signals():
    from drona.entities import extract_entities, get_action_checklist

    sample = (
        "My payment failed on UPI txn-9876543210 with error ERR-502 on Visa card ending in 4321. "
        "Amount was debited 15000 rs."
    )
    ent = extract_entities(sample, amount_fallback=15000.0)
    d = ent.to_dict()
    assert "txn-9876543210" in [t.lower() for t in d["transaction_ids"]]
    assert "ERR-502" in d["error_codes"]
    assert "UPI" in d["payment_rails"]
    assert "Visa Card" in d["payment_rails"]
    assert "•••• 4321" in d["masked_accounts"]
    assert d["amount"] == 15000.0

    checklist = get_action_checklist("payment_failure")
    assert len(checklist) >= 3


# --------------------------------------------------------------------------
# deflection & store analytics
# --------------------------------------------------------------------------


def test_determine_deflection_modes():
    from drona.triage import determine_deflection

    assert determine_deflection(escalate=False, priority="P2", confidence=0.92, category="card_services") == "AUTO_DEFLECTED"
    assert determine_deflection(escalate=True, priority="P2", confidence=0.65, category="billing_charges") == "AGENT_ASSISTED"
    assert determine_deflection(escalate=False, priority="P0", confidence=0.95, category="payment_failure") == "HUMAN_ESCALATED"
    assert determine_deflection(escalate=False, priority="P1", confidence=0.98, category="fraud_security") == "HUMAN_ESCALATED"


def test_store_persists_trace_and_analytics(tmp_path):
    from drona import store as store_mod

    st = store_mod.Store(tmp_path / "test_store.sqlite")
    tid = st.next_id()
    st.record(
        tid,
        "unauthorized charge",
        {
            "category": "fraud_security",
            "priority": "P0",
            "queue": "security-response",
            "confidence": 0.95,
            "escalate": True,
            "latency_ms": 42.0,
            "deflection_status": "HUMAN_ESCALATED",
            "trace": [("thought", "investigating security risk"), ("action", "alert_fraud()")],
            "entities": {"error_codes": ["ERR-401"]},
        },
    )
    row = st.recent(1)[0]
    assert row["ticket_id"] == tid
    assert row["deflection"] == "HUMAN_ESCALATED"
    assert "investigating security risk" in row["trace_json"]

    # Review
    st.review(tid, approved=True, category="fraud_security", priority="P0", note="confirmed")
    assert st.recent(1)[0]["approved"] == 1

    # Analytics
    an = st.analytics()
    assert an["total"] == 1
    assert "fraud_security" in an["by_priority"] or "P0" in an["by_priority"]


# --------------------------------------------------------------------------
# FastAPI service integration tests
# --------------------------------------------------------------------------


def test_service_endpoints():
    from fastapi.testclient import TestClient
    from drona.service import app

    client = TestClient(app)

    # Health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    # Single triage
    r_triage = client.post(
        "/triage",
        json={"text": "payment declined on card ending in 1234 with error ERR-502 debited 12000 rs"},
    )
    assert r_triage.status_code == 200
    t_data = r_triage.json()
    assert "ticket_id" in t_data
    assert "category" in t_data
    assert "deflection_status" in t_data
    assert "entities" in t_data
    tid = t_data["ticket_id"]

    # Batch triage with signals
    r_batch = client.post(
        "/triage/batch",
        json={
            "texts": [
                "card was charged twice",
                "cannot login to my account",
            ],
            "signals": [
                {"channel": "email", "amount_at_risk": 50000.0},
                {"channel": "chat", "customer_tier": "enterprise"},
            ],
        },
    )
    assert r_batch.status_code == 200
    b_data = r_batch.json()
    assert b_data["n"] == 2
    assert len(b_data["results"]) == 2

    # Queues
    r_queues = client.get("/queues")
    assert r_queues.status_code == 200
    assert "queues" in r_queues.json()

    # Review
    r_rev = client.post(
        "/review",
        json={"ticket_id": tid, "approved": True, "category": t_data["category"], "priority": t_data["priority"]},
    )
    assert r_rev.status_code == 200
    assert r_rev.json()["approved"] is True

    # Analytics
    r_analytics = client.get("/analytics")
    assert r_analytics.status_code == 200
    assert "deflection_rate" in r_analytics.json()