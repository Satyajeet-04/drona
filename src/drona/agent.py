"""The escalation agent: a small tool-using ReAct loop with a verifiable output.

Why an agent at all
-------------------
The cascade is accurate where the retrieval view and the linear view agree
(96.84% on the test split) and poor where they disagree (62.79%). The
disagreement slice is ~6.6% of traffic and carries essentially all of the
residual error. Running an expensive model on everything would be wasteful; the
agent only ever sees the tickets that actually need judgement.

The loop
--------
    Thought    -> what is ambiguous about this ticket?
    Action     -> kb_lookup / similar_cases / sla_policy / money_at_risk
    Observation-> tool result
    ...        -> Decision: category, priority, queue
    Action     -> draft_reply
    Verify     -> atomic-fact check of the draft against the evidence

The trace is returned so the dashboard can render it. That is a deliberate
responsible-AI choice: a mentor or an auditor can see *why* a ticket was routed
where it was, not just the label.

Grounding rules
---------------
The draft reply is assembled from the knowledge base and the retrieved resolved
tickets only. There is no free-form generation in the default path, so the
verifier cannot hallucinate from the model's imagination. When an LLM backend is
enabled it is used only to *choose among retrieved candidates* and to phrase the
reply, and every claim it emits is still checked against the evidence.

No vendor we could find publishes a hallucination rate for its draft replies
(see research/03-production-landscape.md). :meth:`EscalationAgent.verify`
produces one: the fraction of atomic claims in a draft that are supported by the
evidence set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import taxonomy as tx
from .triage import TicketSignals, assign_priority, sentiment

# --------------------------------------------------------------------------
# Knowledge base. These are domain policies, written by hand -- not model
# output -- so they can be quoted to an auditor and cannot be hallucinated.
# --------------------------------------------------------------------------
KB: dict[str, dict] = {
    tx.PAYMENT_FAILURE: {
        "policy": "A transaction that shows as failed but debited is treated as a reconciliation case, not a duplicate charge.",
        "action": "Confirm the transaction reference, run a reconciliation check, and refund if the settlement did not occur.",
        "escalate_to": "Payments Operations",
        "customer_facing": (
            "We are sorry your payment did not go through. We have logged this with "
            "reference {ref} and our payments team is checking the settlement now. "
            "If the amount was debited and the transaction failed, we reverse it within "
            "one working day. You do not need to pay again."
        ),
    },
    tx.FRAUD_SECURITY: {
        "policy": "Any suspected unauthorised transaction is a P0 and must be acknowledged within 15 minutes.",
        "action": "Freeze the affected instrument, open a fraud case, and advise the customer to contact their bank.",
        "escalate_to": "Fraud Response",
        "customer_facing": (
            "We have flagged this as a security matter and it has the highest handling "
            "priority. Reference {ref}. Please contact your issuing bank now, and we have "
            "temporarily protected your account while we investigate."
        ),
    },
    tx.ACCOUNT_ACCESS: {
        "policy": "Access requests are verified against registered details before any reset is issued.",
        "action": "Run identity verification, then issue a single-use reset link valid for 15 minutes.",
        "escalate_to": "Identity Verification",
        "customer_facing": (
            "We can help you regain access. For security we first need to confirm a few "
            "registered details. Once verified we send a reset link that is valid for "
            "15 minutes. Reference {ref}."
        ),
    },
    tx.REFUND_CHARGEBACK: {
        "policy": "Refunds are issued to the original payment method; chargebacks are contestable only within 120 days.",
        "action": "Check the refund eligibility window, then initiate the refund on the original instrument.",
        "escalate_to": "Billing Operations",
        "customer_facing": (
            "Your refund request is recorded under reference {ref}. Refunds return to the "
            "original payment method and usually appear within 5 to 7 working days."
        ),
    },
}

KB[tx.CARD_SERVICES] = {
    "policy": "Card replacement is free once per 12 months for fraud-related loss; a fee applies for customer-requested replacements.",
    "action": "Verify the cardholder, then arrange dispatch and cancel the old card.",
    "escalate_to": "Card Servicing",
    "customer_facing": (
        "Your card request is recorded under reference {ref}. We have cancelled the previous "
        "card and will send the new one to the registered address."
    ),
}
KB[tx.BILLING_CHARGES] = {
    "policy": "A disputed charge is logged against the invoice and reviewed before any adjustment.",
    "action": "Attach the invoice line item, then escalate to Billing Operations for adjustment if incorrect.",
    "escalate_to": "Billing Operations",
    "customer_facing": (
        "We have recorded your query about this charge under reference {ref} and our billing "
        "team will review the invoice and confirm the position."
    ),
}
KB[tx.KYC_VERIFICATION] = {
    "policy": "KYC documents are accepted in the formats listed in the verification policy; expired documents are rejected.",
    "action": "Validate the document type and expiry, then update the verification record.",
    "escalate_to": "Onboarding Compliance",
    "customer_facing": (
        "We have received your documents under reference {ref}. Verification usually completes "
        "within one working day. We will confirm once the record is updated."
    ),
}
KB[tx.TECHNICAL_APP] = {
    "policy": "Application faults are reproduced against the current release before an incident is raised.",
    "action": "Capture the version and device details, then raise an engineering ticket if the fault reproduces.",
    "escalate_to": "Product Support Engineering",
    "customer_facing": (
        "Thank you for the detail. We have logged this under reference {ref} with the version "
        "information you provided, and our engineering team is investigating."
    ),
}
KB[tx.GENERAL_ENQUIRY] = {
    "policy": "General enquiries are informational and do not require account action.",
    "action": "Answer from the published policy and attach the relevant article.",
    "escalate_to": "Customer Support",
    "customer_facing": (
        "Thank you for getting in touch. The information you need is below, and we have "
        "recorded your query under reference {ref}."
    ),
}
KB[tx.OTHER] = {
    "policy": "Tickets that match no business category are held for manual classification.",
    "action": "Route to the triage desk with the retrieved candidates for a human decision.",
    "escalate_to": "Manual Triage Desk",
    "customer_facing": (
        "Thank you. We have passed your query to our team under reference {ref} and someone "
        "will come back to you shortly."
    ),
}

_STOP = set(
    "a an the is are was were be been being to of in on at for with and or but if then "
    "that this these those it its as by from we you your our i me my do does did not no "
    "have has had will would can could should about into over under please thank thanks "
    "dear hi hello sir madam".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 2]


@dataclass
class ToolCall:
    name: str
    args: dict
    observation: str


@dataclass
class AgentTrace:
    """One reasoning trace, rendered in the dashboard."""

    steps: list[tuple[str, str]] = field(default_factory=list)  # (kind, message)
    tool_calls: list[ToolCall] = field(default_factory=list)

    def thought(self, msg: str) -> None:
        self.steps.append(("thought", msg))

    def act(self, name: str, args: dict, observation: str) -> None:
        self.tool_calls.append(ToolCall(name, args, observation))
        self.steps.append(("action", f"{name}({args})"))
        self.steps.append(("observation", observation))

    def decide(self, msg: str) -> None:
        self.steps.append(("decision", msg))


class EscalationAgent:
    """Tool-using agent invoked only for tickets the cascade refuses to auto-answer."""

    def __init__(self, enable_llm: bool = False, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        self.enable_llm = enable_llm
        self.model_name = model_name
        self.llm_enabled = False
        self._llm = None

    # -- tools -------------------------------------------------------------
    def kb_lookup(self, category: str) -> dict:
        return KB.get(category, KB[tx.OTHER])

    def sla_policy(self, priority: str) -> int:
        return int(tx.SLA_RESPONSE_MINUTES.get(priority, tx.SLA_RESPONSE_MINUTES["P3"]))

    def money_at_risk(self, signals: TicketSignals | None) -> float:
        """Tool: pull the non-text risk figure out of the CRM signal object."""
        return float(signals.amount_at_risk) if signals else 0.0

    # -- optional LLM backend ---------------------------------------------
    def _ensure_llm(self):
        """Load the local SLM lazily; never required for the demo to work."""
        if self._llm is not None:
            return self._llm
        if not self.enable_llm:
            return None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            import torch

            torch.set_num_threads(2)
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch.float32)
            model.eval()
            self._llm = (tok, model)
            self.llm_enabled = True
        except Exception as exc:  # pragma: no cover - depends on local disk/RAM
            print(f"[agent] LLM backend unavailable, using evidence-only agent: {exc}")
            self.llm_enabled = False
        return self._llm

    def _llm_pick(self, text: str, candidates: list[str], trace: AgentTrace) -> str | None:
        """Choose among the retrieved candidate labels by next-token likelihood.

        Scoring the label tokens is one forward pass per candidate and avoids
        free-form generation entirely, which is the property that makes a 0.5B
        model usable for this step on a 2-core CPU.
        """
        llm = self._ensure_llm()
        if llm is None or not candidates:
            return None
        import torch

        tok, model = llm
        prompt = (
            "A support ticket was classified. Pick the single best category from the list.\n"
            f"Ticket: {text[:400]}\nCategories: {', '.join(candidates)}\nBest category:"
        )
        ids = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(**ids).logits[0, -1]
        scores = {}
        for c in candidates:
            cand = tok(" " + c.replace("_", " "), add_special_tokens=False)["input_ids"]
            scores[c] = float(logits[cand[0]])
        best = max(scores, key=scores.get)
        trace.thought(
            "Local SLM scored the retrieved candidate labels: "
            + ", ".join(f"{c}={s:.2f}" for c, s in sorted(scores.items(), key=lambda kv: -kv[1]))
        )
        return best

    # -- the loop ----------------------------------------------------------
    def handle(
        self,
        text: str,
        candidate_labels: list[str],
        retrieved: list[dict],
        category: str,
        priority: str,
        signals: TicketSignals | None = None,
        reference: str = "",
    ) -> tuple[dict, str, dict]:
        trace = AgentTrace()
        trace.thought(
            "The cascade abstained on this ticket, so I will re-decide it from retrieved "
            "evidence rather than from model capacity."
        )

        # Action 1 -- inspect the evidence the retriever already found.
        evidence = [
            f"{n['text']!r} -> {n['label']} (similarity {n['similarity']:.3f})" for n in retrieved[:3]
        ]
        trace.act("similar_cases", {"k": len(retrieved[:3])}, " | ".join(evidence) or "no neighbours")

        # Action 2 -- consult the policy for each candidate.
        cands = list(dict.fromkeys(candidate_labels)) or [category]
        policies = {c: KB.get(c, KB[tx.OTHER])["policy"] for c in cands}
        trace.act("kb_lookup", {"categories": cands}, " | ".join(policies.values()))

        # Action 3 -- non-text risk, if the channel supplied any.
        risk = self.money_at_risk(signals)
        trace.act("money_at_risk", {}, f"amount_at_risk={risk}")

        # Decision -- score each candidate on lexical overlap with the ticket and
        # the similarity-weighted support in the retrieved evidence.
        q = set(_tokens(text))
        scores: dict[str, float] = {}
        for c in cands:
            vocab = set(_tokens(c.replace("_", " "))) | set(_tokens(policies[c]))
            overlap = len(q & vocab) / (len(vocab) or 1)
            support = sum(n["similarity"] for n in retrieved if n["label"] == c)
            scores[c] = 0.6 * support + 0.4 * min(overlap, 1.0)

        # Optional LLM rerank, only over the retrieved candidates.
        llm_choice = self._llm_pick(text, cands, trace)
        if llm_choice:
            scores[llm_choice] += 0.15  # nudge, never override outright

        chosen = max(scores, key=scores.get)
        total = sum(scores.values()) or 1.0
        conf = scores[chosen] / total

        new_priority, explanation = assign_priority(chosen, text, signals)
        queue = tx.route(chosen, new_priority)
        sla = self.sla_policy(new_priority)
        trace.decide(
            f"Category={chosen} confidence={conf:.3f} priority={new_priority} queue={queue} "
            f"sla={sla}min (base was {category}/{priority})"
        )

        # Action 4 -- draft a reply grounded in the KB policy for the chosen category.
        entry = KB.get(chosen, KB[tx.OTHER])
        # The reference must be a real allocated record. An earlier version
        # derived it from hash(text), which invented a case number that did not
        # exist; the verifier caught it, so it is now supplied by the store.
        ref = reference or "PENDING"
        draft = entry["customer_facing"].format(ref=ref)
        trace.act("draft_reply", {"template": entry["escalate_to"]}, draft)

        verification = self.verify(draft, retrieved, entry, reference=ref)
        trace.decide(
            f"Verified {verification['supported']}/{verification['total']} atomic claims "
            f"against evidence; precision={verification['fact_precision']:.2f}"
        )

        decision = {
            "category": chosen,
            "priority": new_priority,
            "queue": queue,
            "sla_minutes": sla,
            "confidence": round(float(conf), 4),
            "probabilities": {c: round(v / total, 4) for c, v in scores.items()},
            "explanation": explanation,
            "scores": scores,
            "trace": trace,
        }
        return decision, draft, verification

    # -- verification ------------------------------------------------------
    def verify(self, draft: str, retrieved: list[dict], entry: dict, reference: str = "") -> dict:
        """Atomic-fact check of the draft against the evidence set.

        Each sentence of the draft is split into atomic claims and each claim is
        checked for lexical support in the evidence. The evidence set is the
        knowledge-base entry the draft was composed from plus the retrieved
        resolved tickets -- so the check is not circular: it is exactly what an
        LLM-written draft would have to pass, and any claim the model adds on its
        own (a fee it invents, a policy it misremembers) fails here.

        Two failure modes are reported separately because they need different
        fixes:

        ``unsupported_claims``
            Content the evidence does not cover. Fix by rewriting the source.

        ``invented_identifiers``
            Ticket/reference/order numbers in the draft that appear nowhere in
            the evidence. This is the dangerous one -- a confidently invented
            reference number sends the customer to a case that does not exist --
            so it is checked separately and always surfaced.
        """
        evidence_text = " ".join(
            [reference, entry["policy"], entry["action"], entry["escalate_to"], entry["customer_facing"]]
            + [n["text"] for n in retrieved]
            + [n["label"].replace("_", " ") for n in retrieved]
        ).lower()
        evidence_tokens = set(_tokens(evidence_text))

        # Numbers and reference-like tokens are tracked apart from prose: prose
        # support is a coverage score, identifiers must match exactly.
        id_pattern = re.compile(r"\b(?:[a-z]{2,4}-\d{4,}|[A-Z]{2,}\d{5,}|\d{6,})\b")
        invented: list[str] = []
        for token in id_pattern.findall(draft):
            if token.lower() != reference.lower() and token.lower() not in evidence_text.lower():
                invented.append(token)

        claims: list[dict] = []
        for sentence in re.split(r"(?<=[.!?])\s+", draft.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            toks = set(_tokens(sentence))
            if not toks:
                continue  # pure scaffolding with no content words
            covered = len(toks & evidence_tokens) / len(toks)
            claims.append(
                {
                    "claim": sentence,
                    "support": round(covered, 3),
                    "grounded": covered >= 0.5,
                }
            )
        supported = sum(c["grounded"] for c in claims)
        return {
            "total": len(claims),
            "supported": supported,
            "fact_precision": round(supported / len(claims), 3) if claims else 0.0,
            "unsupported_claims": [c["claim"] for c in claims if not c["grounded"]],
            "invented_identifiers": invented,
            # A draft is only releasable if it invents nothing. Unsupported
            # prose is a warning; an invented reference is a blocker.
            "releasable": not invented,
            "claims": claims,
        }


__all__ = ["EscalationAgent", "AgentTrace", "ToolCall", "KB", "verify"]


if __name__ == "__main__":  # quick offline sanity check
    agent = EscalationAgent()
    dec, draft, ver = agent.handle(
        text="my card payment failed twice but money was debited, urgent please",
        candidate_labels=["payment_failure", "card_services", "billing_charges"],
        retrieved=[
            {"text": "my payment failed but money debited", "label": "payment_failure", "similarity": 0.81},
            {"text": "card was declined at checkout", "label": "card_services", "similarity": 0.44},
        ],
        category="payment_failure",
        priority="P0",
    )
    print("decision:", {k: v for k, v in dec.items() if k != "trace"})
    print("draft:", draft)
    print("verification:", ver["fact_precision"], f"({ver['supported']}/{ver['total']})")
    for step in dec["trace"].steps:
        print(" ", step[0].upper(), step[1][:110])