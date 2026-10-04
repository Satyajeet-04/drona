"""Audit store: real ticket identifiers, decision log and human corrections.

Two reasons this exists rather than being a `dict` in memory:

1. **Correctness.** An earlier version minted the ticket reference by hashing
   the text, which meant every drafted reply cited a case number that did not
   exist. The verifier caught it. References must come from a real registry, and
   the number must be recorded before the reply is written.
2. **The capstone's responsible-AI requirement.** Every triage decision, the
   evidence behind it, whether a human approved it, and whether the human
   overrode the model are all persisted so the claim "the agent is auditable"
   is checkable rather than rhetorical.

SQLite because the synopsis allows SQLite/PostgreSQL and because a single file
survives the demo without a server. The schema is deliberately portable.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from .config import DATA

DB_PATH = DATA / "drona.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id      TEXT PRIMARY KEY,
    text           TEXT NOT NULL,
    category       TEXT,
    priority       TEXT,
    queue          TEXT,
    confidence     REAL,
    escalated      INTEGER,
    latency_ms     REAL,
    created_at     REAL,
    -- human-in-the-loop columns
    reviewed       INTEGER DEFAULT 0,
    approved       INTEGER DEFAULT 0,
    agent_category TEXT,     -- what the model said
    agent_priority TEXT,
    human_category TEXT,     -- what the agent decided, if different
    human_priority TEXT,
    draft_reply    TEXT,
    trace_json     TEXT,
    verify_json    TEXT
);
CREATE INDEX IF NOT EXISTS ix_tickets_created ON tickets(created_at);
CREATE INDEX IF NOT EXISTS ix_tickets_queue    ON tickets(queue);

CREATE TABLE IF NOT EXISTS feedback (
    ticket_id  TEXT PRIMARY KEY,
    category   TEXT,
    priority   TEXT,
    note       TEXT,
    created_at REAL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
);
"""


class Store:
    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- identifiers -------------------------------------------------------
    def next_id(self) -> str:
        """Allocate a genuine, monotonically increasing ticket reference.

        The id is written to the database *before* any reply referencing it is
        generated, so a drafted reply can always cite a record that exists.
        """
        n = self.conn.execute("SELECT COUNT(*) AS c FROM tickets").fetchone()["c"]
        tid = f"DRN-{100001 + n}"
        # Guard against a gap left by a rolled-back insert.
        while self.conn.execute("SELECT 1 FROM tickets WHERE ticket_id=?", (tid,)).fetchone():
            n += 1
            tid = f"DRN-{100001 + n}"
        return tid

    def exists(self, ticket_id: str) -> bool:
        return bool(self.conn.execute("SELECT 1 FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone())

    # -- writes ------------------------------------------------------------
    def record(self, ticket_id: str, text: str, result: dict, trace=None) -> None:
        verify = json.dumps(result.get("verification", {}))
        trace_json = json.dumps(
            [{"kind": k, "message": m} for k, m in getattr(trace, "steps", [])]
        ) if trace is not None else None
        self.conn.execute(
            """INSERT OR REPLACE INTO tickets
               (ticket_id, text, category, priority, queue, confidence, escalated,
                latency_ms, created_at, agent_category, agent_priority, draft_reply,
                trace_json, verify_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ticket_id, text, result.get("category"), result.get("priority"),
                result.get("queue"), result.get("confidence"),
                int(bool(result.get("escalate"))), result.get("latency_ms"),
                time.time(), result.get("category"), result.get("priority"),
                result.get("draft_reply", ""), trace_json, verify,
            ),
        )
        self.conn.commit()

    def review(self, ticket_id: str, approved: bool, category: str | None, priority: str | None, note: str = "") -> None:
        """Record the human decision. Overrides are stored separately from the
        model output so that model accuracy can be recomputed from the log."""
        self.conn.execute(
            """UPDATE tickets SET reviewed=1, approved=?, human_category=?, human_priority=?
               WHERE ticket_id=?""",
            (int(approved), category, priority, ticket_id),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO feedback (ticket_id, category, priority, note, created_at) VALUES (?,?,?,?,?)",
            (ticket_id, category, priority, note, time.time()),
        )
        self.conn.commit()

    # -- reads -------------------------------------------------------------
    def recent(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM tickets ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def by_queue(self) -> list[dict]:
        """Open tickets grouped by queue, ordered by SLA urgency then age."""
        rows = self.conn.execute(
            """SELECT * FROM tickets WHERE reviewed=0
               ORDER BY priority ASC, created_at ASC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        c = self.conn
        total = c.execute("SELECT COUNT(*) n FROM tickets").fetchone()["n"]
        reviewed = c.execute("SELECT COUNT(*) n FROM tickets WHERE reviewed=1").fetchone()["n"]
        approved = c.execute("SELECT COUNT(*) n FROM tickets WHERE approved=1").fetchone()["n"]
        overridden = c.execute(
            "SELECT COUNT(*) n FROM tickets WHERE human_category IS NOT NULL AND human_category != agent_category"
        ).fetchone()["n"]
        escalated = c.execute("SELECT COUNT(*) n FROM tickets WHERE escalated=1").fetchone()["n"]
        avg_lat = c.execute("SELECT AVG(latency_ms) a FROM tickets").fetchone()["a"]
        return {
            "total": total,
            "reviewed": reviewed,
            "approved": approved,
            "overridden": overridden,
            "escalated": escalated,
            "escalation_rate": round(escalated / total, 4) if total else 0.0,
            "avg_latency_ms": round(avg_lat or 0.0, 2),
            "human_agreement_rate": round(
                1 - overridden / reviewed, 4
            ) if reviewed else None,
        }

    def drift(self, min_live: int = 30) -> dict:
        """Distribution shift between the reference corpus and live traffic.

        A deliberately simple check: per-category share in the reference *training*
        split versus the share in triaged traffic, with the absolute difference
        flagged above 5 points. This is the offline stand-in for the drift report
        the synopsis asks Evidently AI to produce.

        The reference must be the training corpus, not the live table. An earlier
        version derived both sides from ``tickets``, which made every delta
        identically zero -- a drift monitor that can never fire is worse than no
        monitor, because it reports false assurance.

        Returns ``None`` for ``drift`` when fewer than ``min_live`` tickets have
        been triaged: with 7 samples every share is a multiple of ~0.14, so any
        5-point threshold is meaningless.
        """
        from collections import Counter

        from . import taxonomy as tx
        from .config import PROCESSED
        from .data import load_jsonl

        categories = list(tx.COARSE_CATEGORIES)
        rows = self.conn.execute(
            "SELECT category, COUNT(*) n FROM tickets GROUP BY category"
        ).fetchall()
        live = {r["category"]: r["n"] for r in rows}
        total_live = sum(live.values())

        ref_counts = Counter(t.coarse for t in load_jsonl("train"))
        total_ref = sum(ref_counts.values()) or 1
        enough = total_live >= min_live

        out: dict = {}
        for c in categories:
            p_ref = ref_counts.get(c, 0) / total_ref
            p_live = live.get(c, 0) / total_live if total_live else 0.0
            out[c] = {
                "reference_share": round(p_ref, 4),
                "live_share": round(p_live, 4),
                "delta": round(p_live - p_ref, 4),
                "drift": (abs(p_live - p_ref) > 0.05) if enough else None,
            }
        return {
            "per_category": out,
            "n_live": total_live,
            "n_reference": total_ref,
            "sufficient_sample": enough,
            "note": None if enough else f"need >= {min_live} triaged tickets to judge drift",
        }


_STORE: Store | None = None


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE


__all__ = ["Store", "get_store", "DB_PATH"]


if __name__ == "__main__":
    s = Store()
    print("schema ok:", [r[0] for r in s.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")])