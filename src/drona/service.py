"""FastAPI service for the DRONA engine.

Endpoints
---------
``POST /triage``          triage one ticket, returning category, priority, queue,
                          evidence, the agent trace (when escalated) and a
                          grounded draft reply with its verification report.
``POST /triage/batch``    triage many tickets, using the vectorised retrieval path.
``GET  /queues``          the agent work-queue: open tickets ordered by SLA.
``POST /review``          human-in-the-loop decision: approve, or correct.
``GET  /metrics``         operational counters: escalation rate, latency, agreement.
``GET  /drift``           reference-vs-live category distribution shift.
``GET  /health``          liveness plus engine readiness.

The service is stateless apart from the SQLite audit store, so it starts in
about a second and holds no model state beyond the fitted cascade. It makes no
outbound network calls: the demo behaves identically offline.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import get_engine
from .store import get_store

app = FastAPI(
    title="DRONA Smart Ticket Understanding Engine",
    version="1.0",
    description=(
        "Offline, CPU-only ticket triage: retrieval + linear fusion with a "
        "calibrated abstention gate, SLA priority routing, and a tool-using "
        "agent that re-decides only the tickets the cascade refuses."
    ),
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class Signals(BaseModel):
    channel: str = Field("web", description="web | email | chat | phone")
    customer_tier: str = Field("standard", description="standard | premier | enterprise")
    amount_at_risk: float = 0.0
    sla_deadline_minutes: float | None = None


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw ticket text")
    signals: Signals | None = None


class BatchRequest(BaseModel):
    texts: list[str]
    signals: list[Signals] | None = None


class ReviewRequest(BaseModel):
    ticket_id: str
    approved: bool
    category: str | None = None
    priority: str | None = None
    note: str = ""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _as_dict(result) -> dict:
    d = asdict(result)
    return d


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    engine = get_engine()
    return {"status": "ok", "ready": engine.ready, **engine.metrics()}


@app.post("/triage")
def triage(req: TriageRequest) -> dict:
    engine = get_engine()
    store = get_store()
    ticket_id = store.next_id()
    signals = req.signals.model_dump() if req.signals else None
    result = engine.triage(req.text, signals, ticket_id=ticket_id)
    store.record(ticket_id, req.text, _as_dict(result))
    return _as_dict(result)


@app.post("/triage/batch")
def triage_batch(req: BatchRequest) -> dict:
    engine = get_engine()
    store = get_store()
    results = []
    for text in req.texts:
        # Allocate a real audit id per ticket. Without this the batch path
        # returned empty ticket_ids, which broke traceability back to the
        # SQLite store and made batch results unreviewable via /review.
        ticket_id = store.next_id()
        result = engine.triage(text, ticket_id=ticket_id)
        store.record(ticket_id, text, _as_dict(result))
        results.append(_as_dict(result))
    return {"n": len(results), "results": results}


@app.get("/queues")
def queues() -> dict:
    store = get_store()
    tickets = store.by_queue()
    grouped: dict[str, list] = {}
    for t in tickets:
        grouped.setdefault(t["queue"] or "unrouted", []).append(t)
    return {
        "queues": {k: len(v) for k, v in grouped.items()},
        "tickets": grouped,
    }


@app.post("/review")
def review(req: ReviewRequest) -> dict:
    store = get_store()
    if not store.exists(req.ticket_id):
        raise HTTPException(status_code=404, detail=f"unknown ticket {req.ticket_id}")
    store.review(req.ticket_id, req.approved, req.category, req.priority, req.note)
    return {"ticket_id": req.ticket_id, "reviewed": True, "approved": req.approved}


@app.get("/metrics")
def metrics() -> dict:
    store = get_store()
    return {"engine": get_engine().metrics(), "operations": store.stats()}


@app.get("/drift")
def drift() -> dict:
    return get_store().drift()


if __name__ == "__main__":
    import uvicorn

    # Bind 0.0.0.0, not 127.0.0.1: on a loopback bind the API is invisible to
    # anyone opening http://<this-host>:8000 from another machine, which reads
    # as "the project does not run". Inbound-only -- nothing leaves the host.
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")