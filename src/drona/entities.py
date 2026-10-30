"""Context and entity extraction for support tickets.

Extracts structured operational entities from unstructured ticket prose,
emails, and chat transcripts:
  * Transaction / Reference IDs (TXN, REF, UPI, UTR, 12-digit references)
  * Error codes (e.g. ERR-502, DECLINED_102, HTTP 404/500, AUTH_TIMEOUT)
  * Payment rails & instruments (UPI, NEFT, RTGS, IMPS, Visa, Mastercard, RuPay)
  * Masked account & card identifiers (PII-safe)
  * Monetary amounts (prefix and postfix Indian & international formats)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Regex patterns
# --------------------------------------------------------------------------
_TXN_PATTERN = re.compile(
    r"\b(?:"
    r"txn[-_]?[a-z0-9]{6,}|"
    r"ref[-_]?[a-z0-9]{6,}|"
    r"upi[-_]?[a-z0-9]{8,}|"
    r"utr[-_]?[a-z0-9]{8,}|"
    r"[0-9]{12}"
    r")\b",
    re.IGNORECASE,
)

_ERROR_CODE_PATTERN = re.compile(
    r"\b(?:"
    r"err(?:or)?[-_]?[0-9]{3,4}|"
    r"code[-_]?[0-9]{3,4}|"
    r"declined[-_]?[0-9]{2,4}|"
    r"auth[-_]?(?:failed|timeout|declined)|"
    r"http[-_]?[45][0-9]{2}|"
    r"status[-_]?[45][0-9]{2}|"
    r"[45][0-9]{2}\s+error"
    r")\b",
    re.IGNORECASE,
)

_MASKED_ACCOUNT_PATTERN = re.compile(
    r"(?:(?:\*{4}|[•]{4}|x{4})\s*[\s-]?\s*(\d{4})|\b(?:ending\s+in|a/c\s+no\.?)\s+(\d{4})\b)",
    re.IGNORECASE,
)

_RAILS_LEXICON = {
    "upi": "UPI",
    "gpay": "Google Pay (UPI)",
    "phonepe": "PhonePe (UPI)",
    "paytm": "Paytm",
    "neft": "NEFT",
    "rtgs": "RTGS",
    "imps": "IMPS",
    "visa": "Visa Card",
    "mastercard": "Mastercard",
    "rupay": "RuPay Card",
    "maestro": "Maestro Card",
    "net banking": "Net Banking",
    "netbanking": "Net Banking",
    "credit card": "Credit Card",
    "debit card": "Debit Card",
    "atm": "ATM",
}

ACTION_CHECKLISTS: dict[str, list[str]] = {
    "payment_failure": [
        "Check payment gateway switch logs for gateway response code",
        "Verify dual-inquiry / reconciliation status with beneficiary bank",
        "If settlement failed, trigger auto-refund reversal (SLA: T+1 day)",
        "Confirm masked account/card details with customer before re-initiating",
    ],
    "fraud_security": [
        "IMMEDIATE: Freeze affected card / digital instrument to prevent further loss",
        "File suspicious transaction report (STR) in compliance console",
        "Check IP/device footprint against high-risk geolocation blacklist",
        "Advise customer to contact issuing bank and register incident acknowledgement",
    ],
    "account_access": [
        "Validate registered phone/email via 2FA out-of-band verification",
        "Check account freeze or suspicious login alert flags",
        "Dispatch 15-minute time-bound secure password reset link",
        "Advise user to update security questions upon access restoration",
    ],
    "refund_chargeback": [
        "Verify transaction age against 120-day chargeback contestability window",
        "Match merchant ARN (Acquirer Reference Number) with clearing record",
        "Initiate chargeback dispute via card network portal (Visa/Mastercard/RuPay)",
        "Issue acknowledgement with expected 5-7 working days credit timeline",
    ],
    "card_services": [
        "Validate active card status (active, damaged, blocked, or expired)",
        "Check fee eligibility (free replacement for compromised/fraud cards)",
        "Dispatch replacement card to KYC-verified residential address",
        "Instruct customer to generate new PIN via mobile banking upon delivery",
    ],
    "billing_charges": [
        "Retrieve detailed fee schedule and monthly billing statement breakdown",
        "Cross-reference levied surcharge or annual maintenance charge (AMC)",
        "If fee was improperly debited, apply ledger credit adjustment",
        "Send itemized fee clarification note to customer",
    ],
    "kyc_verification": [
        "Check submitted document format and validity against regulatory guidelines",
        "Verify Aadhaar / PAN / Passport OCR match with core banking database",
        "Confirm selfie liveness check and address verification status",
        "Update KYC status flag in customer master profile (SLA: 24h)",
    ],
    "technical_app": [
        "Check reported app version, OS build, and device manufacturer",
        "Inspect application APM and crashlytics error logs",
        "Advise user to clear app cache, check biometric permissions, or reinstall",
        "If systemic, escalate ticket with reproduction steps to Core Engineering",
    ],
    "general_enquiry": [
        "Search knowledge base repository for verified policy article",
        "Provide standardized informational guidance to the customer",
        "Attach relevant banking product / schedule of charges documentation",
        "Close ticket with feedback rating request",
    ],
    "other": [
        "Route ticket to manual triage supervisor for specialist evaluation",
        "Check whether customer intent relates to loans, wealth, or NRI banking",
    ],
}


@dataclass
class ExtractedEntities:
    """Entities discovered in the ticket prose or communication channel."""

    transaction_ids: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)
    payment_rails: list[str] = field(default_factory=list)
    masked_accounts: list[str] = field(default_factory=list)
    amount: float = 0.0

    def to_dict(self) -> dict:
        return {
            "transaction_ids": self.transaction_ids,
            "error_codes": self.error_codes,
            "payment_rails": self.payment_rails,
            "masked_accounts": self.masked_accounts,
            "amount": self.amount,
        }


def extract_entities(text: str, amount_fallback: float = 0.0) -> ExtractedEntities:
    """Parse text and extract all operational domain entities."""
    low = text.lower()

    txns = list(dict.fromkeys(_TXN_PATTERN.findall(text)))
    errs = list(dict.fromkeys(_ERROR_CODE_PATTERN.findall(text)))

    rails: list[str] = []
    for term, label in _RAILS_LEXICON.items():
        if re.search(r"\b" + re.escape(term) + r"\b", low):
            if label not in rails:
                rails.append(label)

    masked: list[str] = []
    for m in _MASKED_ACCOUNT_PATTERN.finditer(text):
        last4 = m.group(1) or m.group(2)
        if last4:
            masked.append(f"•••• {last4}")
    masked = list(dict.fromkeys(masked))

    return ExtractedEntities(
        transaction_ids=txns,
        error_codes=[e.upper() for e in errs],
        payment_rails=rails,
        masked_accounts=masked,
        amount=amount_fallback,
    )


def get_action_checklist(category: str) -> list[str]:
    """Return recommended next-step actions for the resolver group."""
    return ACTION_CHECKLISTS.get(category, ACTION_CHECKLISTS["other"])
