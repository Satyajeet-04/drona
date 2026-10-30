"""Priority assignment, routing and the escalation decision.

The synopsis describes a "SLA Priority Engine" that combines category, sentiment
and urgency. In the published literature there is no agreed way to turn text into
an ITIL P0-P3 label -- the method is proprietary at every vendor we could find.
So rather than assert a rule table, this module makes the rules explicit and
auditable, and records every feature that contributed to the decision so an
agent can justify it.

Two design points worth stating plainly:

* **Priority is never text-only.** Production systems fuse text with channel,
  customer tier, transaction amount and SLA proximity (see
  research/03-production-landscape.md). Here, ``TicketSignals`` carries those
  non-text inputs explicitly; when they are absent the engine degrades to
  text-only and says so.
* **Category impact sets the ceiling, urgency raises it.** A fraud alert is
  P0-worthy because of what the category is, not because the customer used an
  angry word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .taxonomy import CATEGORY_IMPACT, PRIORITIES, SLA_RESPONSE_MINUTES, route

# Lexicons are deliberately small and auditable. Every keyword that can change a
# priority is listed here so the review can read the whole policy.
URGENCY_TERMS: dict[str, int] = {
    # Security / money-loss signals -> immediate
    "unauthorized": 3, "fraud": 3, "stolen": 3, "hacked": 3, "scam": 3,
    "compromised": 3, "blocked": 2, "locked out": 2, "cannot access": 2,
    "money debited": 3, "double charged": 3, "wrong amount": 2, "missing money": 3,
    "urgent": 2, "immediately": 2, "asap": 2, "emergency": 3, "critical": 3,
    "not working": 1, "failed": 1, "declined": 2, "rejected": 2, "pending": 1,
    "escalate": 2, "legal": 3, "police": 3, "court": 3, "regulator": 3,
    "gdpr": 2, "complaint": 2, "compensation": 2,
}

NEGATIVE_TERMS = (
    "angry", "furious", "unacceptable", "terrible", "awful", "worst",
    "disappointed", "ridiculous", "frustrated", "useless", "scam",
    # Banking-complaint vocabulary. Without these the analyser called
    # "payment failed 3 times, money debited but not credited" *neutral*,
    # which under-reports urgency on exactly the tickets that matter most.
    "failed", "not credited", "has not been credited", "missing money",
    "no response", "no reply", "still waiting", "not resolved", "unresolved",
    "helpless", "worst experience", "very bad", "not working", "declined",
    "pending since", "since morning", "multiple times", "again and again",
    "third time", "second time", "escalat",
)

# Currency symbols / words that mark a number as money at risk.
_CURRENCY = r"(?:rs\.?|inr|₹|rupees?|\$|usd)\s*"
_POST_CURRENCY = r"\s*(?:rs\.?|inr|₹|rupees?|usd)\b"
# Indian and international digit grouping: 45,000 / 1,50,000 / 1,000,000.
# The grouped branch is listed FIRST and requires at least one comma group, so
# "Rs 45000" cannot be truncated to "450" by the shorter alternative.
_NUMBER = r"(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?"
_LAKH = r"(\d+(?:\.\d+)?)\s*(?:lacs?|lakhs?)"
_CRORE = r"(\d+(?:\.\d+)?)\s*(?:crores?|cr)"
_K_SUFFIX = r"(\d+(?:\.\d+)?)\s*k(?![a-z])"


def extract_amount(text: str) -> float:
    """Best-effort money-at-risk extraction from free ticket text.

    A production system would read the amount from the payment gateway, but
    most inbound tickets arrive as prose, so the amount has to be recovered from
    the body. Supports Indian digit grouping (1,50,000), lakh/crore words, the
    ``45k`` shorthand, and bare `Rs`/`₹`/`INR`/`$` prefixes as well as postfixes.
    Returns the **largest** amount found.
    """
    low = text.lower()
    amounts: list[float] = []

    for m in re.finditer(_CRORE, low):
        amounts.append(float(m.group(1)) * 10_000_000)
    for m in re.finditer(_LAKH, low):
        amounts.append(float(m.group(1)) * 100_000)
    for m in re.finditer(_K_SUFFIX, low):
        amounts.append(float(m.group(1)) * 1_000)
    for m in re.finditer(_CURRENCY + r"(" + _NUMBER + r")", low):
        try:
            amounts.append(float(m.group(1).replace(",", "")))
        except ValueError:
            continue
    for m in re.finditer(r"(" + _NUMBER + r")" + _POST_CURRENCY, low):
        try:
            amounts.append(float(m.group(1).replace(",", "")))
        except ValueError:
            continue

    return max(amounts) if amounts else 0.0


@dataclass
class TicketSignals:
    """Non-text context that a real triage system would receive from CRM/IVR."""

    channel: str = "web"  # web | email | chat | phone
    customer_tier: str = "standard"  # standard | premier | enterprise
    amount_at_risk: float = 0.0  # rupees
    sla_deadline_minutes: float | None = None  # minutes until the clock stops

    def provided(self) -> bool:
        return (self.channel, self.customer_tier, self.amount_at_risk) != (
            "web",
            "standard",
            0.0,
        )


def _hits(text: str, lexicon: dict[str, int]) -> tuple[int, list[str]]:
    low = text.lower()
    score = 0
    matched = []
    for term, weight in lexicon.items():
        if term in low:
            score += weight
            matched.append(term)
    return score, matched


def sentiment(text: str) -> tuple[str, list[str]]:
    """Rule-based sentiment. Deliberately transparent -- a learned sentiment
    model adds latency and a black box for very little on this vocabulary."""
    low = text.lower()
    neg = [t for t in NEGATIVE_TERMS if t in low]
    if len(neg) >= 2:
        return "negative", neg
    if len(neg) == 1:
        return "neutral", neg
    return "neutral", []


def assign_priority(
    category: str,
    text: str,
    signals: TicketSignals | None = None,
) -> tuple[str, dict]:
    """Return ``(priority, explanation)``.

    The explanation is a dict rather than a string so the dashboard can render
    each contributing signal as its own row -- the responsible-AI requirement
    from the Drona 3.0 curriculum, and the thing production vendors never show.
    """
    signals = signals or TicketSignals()
    urgency_score, urgent_terms = _hits(text, URGENCY_TERMS)
    sentiment_label, neg_terms = sentiment(text)

    impact = CATEGORY_IMPACT.get(category, 2)

    # Money at risk: prefer the authoritative value from CRM/payment-gateway
    # signals, else recover it from the ticket prose. `amount_source` records
    # which was used so the audit trail never hides a guess.
    amount = signals.amount_at_risk
    amount_source = "signal"
    if amount <= 0:
        amount = extract_amount(text)
        amount_source = "text_extracted" if amount > 0 else "none"

    reasons: dict = {
        "category": category,
        "category_impact": impact,
        "urgency_score": urgency_score,
        "urgency_terms": urgent_terms,
        "sentiment": sentiment_label,
        "sentiment_terms": neg_terms,
        "channel": signals.channel,
        "customer_tier": signals.customer_tier,
        "amount_at_risk": amount,
        "amount_source": amount_source,
        "non_text_signals_used": signals.provided(),
    }

    # Base priority from impact: 4 -> P0, 3 -> P1, 2 -> P2, 1 -> P3.
    base = {4: "P0", 3: "P1", 2: "P2", 1: "P3"}[impact]

    # Urgency can escalate but never de-escalate below the category floor.
    escalated = None
    if urgency_score >= 3 and base in ("P2", "P3"):
        escalated = "P1"
    elif urgency_score >= 1 and base == "P3":
        escalated = "P2"
    if sentiment_label == "negative" and base == "P3":
        escalated = escalated or "P2"

    # Non-text signals: money at risk or an imminent SLA deadline escalates.
    if amount >= 100_000 and base in ("P2", "P3"):
        escalated = "P1" if escalated in (None, "P3") else escalated
        reasons["escalated_by"] = "amount_at_risk"
    if signals.customer_tier == "enterprise" and base == "P3":
        escalated = escalated or "P2"
    if signals.sla_deadline_minutes is not None and signals.sla_deadline_minutes < 30:
        escalated = "P0" if impact >= 3 else (escalated or "P1")
        reasons["escalated_by"] = "sla_deadline"

    priority = escalated or base
    reasons["base_priority"] = base
    reasons["final_priority"] = priority
    reasons["sla_minutes"] = SLA_RESPONSE_MINUTES[priority]
    return priority, reasons


def determine_deflection(escalate: bool, priority: str, confidence: float, category: str) -> str:
    """Classify resolution mode: AUTO_DEFLECTED | AGENT_ASSISTED | HUMAN_ESCALATED."""
    if category == "fraud_security" or priority == "P0":
        return "HUMAN_ESCALATED"
    if escalate or confidence < 0.7635:
        return "AGENT_ASSISTED"
    return "AUTO_DEFLECTED"


@dataclass
class TriageResult:
    """Everything the API, the dashboard and the audit log need."""

    text: str
    ticket_id: str
    category: str
    confidence: float
    probabilities: dict[str, float]
    priority: str
    queue: str
    sla_minutes: int
    sentiment: str
    escalate: bool
    escalated_because: str = ""
    explanation: dict = field(default_factory=dict)
    neighbours: list[dict] = field(default_factory=list)
    draft_reply: str = ""
    verification: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    trace: list = field(default_factory=list)
    agent_proposal: dict | None = None
    # Hackathon & enterprise extensions:
    deflection_status: str = "AUTO_DEFLECTED"  # AUTO_DEFLECTED | AGENT_ASSISTED | HUMAN_ESCALATED
    entities: dict = field(default_factory=dict)
    action_checklist: list[str] = field(default_factory=list)


__all__ = [
    "TicketSignals",
    "TriageResult",
    "assign_priority",
    "sentiment",
    "extract_amount",
    "determine_deflection",
    "URGENCY_TERMS",
    "PRIORITIES",
    "route",
]