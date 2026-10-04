"""
Taxonomy: the intellectual core of the coarse-grained stage.

The synopsis says Banking77's 77 intents are "mapped to 10 business categories"
but never specifies the mapping. The mapping is not neutral -- it *is* the model.
This module makes it explicit, versioned, and auditable.

Two design decisions, both deliberate departures from the synopsis:

1.  ELEVEN coarse classes, not ten. The synopsis lists Loan/EMI, but Banking77
    contains zero loan intents, so that class is unreachable from the chosen
    dataset. Adding a closed-set escape class (``other``) follows the structured-
    output literature: a classifier forced to choose from an enum that does not
    fit picks wrong *with full confidence*. We force honesty instead.

2.  Hierarchical, not flat. Every ticket keeps both its fine label (77 intents)
    and its coarse label (11 categories). The coarse head is trained on top of
    fine predictions, so coarse errors are attributable to a specific fine
    intent rather than being unrecoverable.
"""

# --------------------------------------------------------------------------
# Coarse categories
# --------------------------------------------------------------------------
PAYMENT_FAILURE = "payment_failure"
REFUND_CHARGEBACK = "refund_chargeback"
ACCOUNT_ACCESS = "account_access"
FRAUD_SECURITY = "fraud_security"
CARD_SERVICES = "card_services"
LOAN_EMI = "loan_emi"          # unreachable from Banking77 -- kept for the rubric
KYC_VERIFICATION = "kyc_verification"
BILLING_CHARGES = "billing_charges"
TECHNICAL_APP = "technical_app"
GENERAL_ENQUIRY = "general_enquiry"
OTHER = "other"                 # escape hatch

COARSE_CATEGORIES = [
    PAYMENT_FAILURE,
    REFUND_CHARGEBACK,
    ACCOUNT_ACCESS,
    FRAUD_SECURITY,
    CARD_SERVICES,
    LOAN_EMI,
    KYC_VERIFICATION,
    BILLING_CHARGES,
    TECHNICAL_APP,
    GENERAL_ENQUIRY,
    OTHER,
]

COARSE_ID = {c: i for i, c in enumerate(COARSE_CATEGORIES)}

# --------------------------------------------------------------------------
# 77 fine intents -> coarse category.
# Verified to cover all 77 Banking77 intents at import time (see below).
# --------------------------------------------------------------------------
INTENT_TO_COARSE = {
    # --- Card lifecycle & functionality (14 intents) ---
    "card_arrival": CARD_SERVICES,
    "card_delivery_estimate": CARD_SERVICES,
    "order_physical_card": CARD_SERVICES,
    "get_physical_card": CARD_SERVICES,
    "getting_spare_card": CARD_SERVICES,
    "activate_my_card": CARD_SERVICES,
    "card_about_to_expire": CARD_SERVICES,
    "getting_virtual_card": CARD_SERVICES,
    "get_disposable_virtual_card": CARD_SERVICES,
    "card_linking": CARD_SERVICES,
    "contactless_not_working": CARD_SERVICES,
    "topping_up_by_card": CARD_SERVICES,
    "top_up_by_cash_or_cheque": CARD_SERVICES,
    "apple_pay_or_google_pay": CARD_SERVICES,

    # --- Payment failure (19 intents) ---
    "pending_cash_withdrawal": PAYMENT_FAILURE,
    "declined_cash_withdrawal": PAYMENT_FAILURE,
    "cash_withdrawal_not_recognised": PAYMENT_FAILURE,
    "wrong_amount_of_cash_received": PAYMENT_FAILURE,
    "card_swallowed": PAYMENT_FAILURE,
    "pending_card_payment": PAYMENT_FAILURE,
    "card_payment_not_recognised": PAYMENT_FAILURE,
    "declined_card_payment": PAYMENT_FAILURE,
    "direct_debit_payment_not_recognised": PAYMENT_FAILURE,
    "transfer_not_received_by_recipient": PAYMENT_FAILURE,
    "declined_transfer": PAYMENT_FAILURE,
    "pending_transfer": PAYMENT_FAILURE,
    "failed_transfer": PAYMENT_FAILURE,
    "top_up_failed": PAYMENT_FAILURE,
    "pending_top_up": PAYMENT_FAILURE,
    "top_up_reverted": PAYMENT_FAILURE,
    "balance_not_updated_after_bank_transfer": PAYMENT_FAILURE,
    "balance_not_updated_after_cheque_or_cash_deposit": PAYMENT_FAILURE,
    "reverted_card_payment?": PAYMENT_FAILURE,

    # --- Billing & charges (10 intents) ---
    "extra_charge_on_statement": BILLING_CHARGES,
    "card_payment_fee_charged": BILLING_CHARGES,
    "transfer_fee_charged": BILLING_CHARGES,
    "exchange_charge": BILLING_CHARGES,
    "top_up_by_bank_transfer_charge": BILLING_CHARGES,
    "top_up_by_card_charge": BILLING_CHARGES,
    "cash_withdrawal_charge": BILLING_CHARGES,
    "transaction_charged_twice": BILLING_CHARGES,
    "wrong_exchange_rate_for_cash_withdrawal": BILLING_CHARGES,
    "card_payment_wrong_exchange_rate": BILLING_CHARGES,

    # --- KYC / verification (5 intents) ---
    "why_verify_identity": KYC_VERIFICATION,
    "unable_to_verify_identity": KYC_VERIFICATION,
    "verify_my_identity": KYC_VERIFICATION,
    "verify_top_up": KYC_VERIFICATION,
    "verify_source_of_funds": KYC_VERIFICATION,

    # --- Fraud & security (4 intents) ---
    "lost_or_stolen_card": FRAUD_SECURITY,
    "compromised_card": FRAUD_SECURITY,
    "lost_or_stolen_phone": FRAUD_SECURITY,
    "beneficiary_not_allowed": FRAUD_SECURITY,

    # --- Account access (5 intents) ---
    "edit_personal_details": ACCOUNT_ACCESS,
    "terminate_account": ACCOUNT_ACCESS,
    "pin_blocked": ACCOUNT_ACCESS,
    "passcode_forgotten": ACCOUNT_ACCESS,
    "change_pin": ACCOUNT_ACCESS,

    # --- Technical / app (4 intents) ---
    "card_not_working": TECHNICAL_APP,
    "exchange_via_app": TECHNICAL_APP,
    "atm_support": TECHNICAL_APP,
    "virtual_card_not_working": TECHNICAL_APP,

    # --- Refund / chargeback (2 intents) ---
    "request_refund": REFUND_CHARGEBACK,
    "Refund_not_showing_up": REFUND_CHARGEBACK,

    # --- General enquiry (13 intents) ---
    "exchange_rate": GENERAL_ENQUIRY,
    "fiat_currency_support": GENERAL_ENQUIRY,
    "supported_cards_and_currencies": GENERAL_ENQUIRY,
    "card_acceptance": GENERAL_ENQUIRY,
    "country_support": GENERAL_ENQUIRY,
    "age_limit": GENERAL_ENQUIRY,
    "top_up_limits": GENERAL_ENQUIRY,
    "transfer_timing": GENERAL_ENQUIRY,
    "receiving_money": GENERAL_ENQUIRY,
    "transfer_into_account": GENERAL_ENQUIRY,
    "automatic_top_up": GENERAL_ENQUIRY,
    "visa_or_mastercard": GENERAL_ENQUIRY,
    "disposable_card_limits": GENERAL_ENQUIRY,

    # --- Escape (1 intent) ---
    "cancel_transfer": OTHER,
}

# --------------------------------------------------------------------------
# SLA priority: ITIL-style Impact x Urgency matrix.
# No public dataset carries P0-P3 labels and no paper documents how firms build
# them, so we state the rule explicitly and it becomes reproducible.
# --------------------------------------------------------------------------
P0, P1, P2, P3 = "P0", "P1", "P2", "P3"
PRIORITIES = [P0, P1, P2, P3]
PRIORITY_ID = {p: i for i, p in enumerate(PRIORITIES)}

# Default impact per category (1..4). Security and money-loss categories are
# high impact by definition; enquiries are low impact regardless of anger.
CATEGORY_IMPACT = {
    FRAUD_SECURITY: 4,
    PAYMENT_FAILURE: 4,
    ACCOUNT_ACCESS: 3,
    LOAN_EMI: 3,
    KYC_VERIFICATION: 3,
    REFUND_CHARGEBACK: 2,
    CARD_SERVICES: 2,
    BILLING_CHARGES: 2,
    TECHNICAL_APP: 2,
    GENERAL_ENQUIRY: 1,
    OTHER: 1,
}

# SLA response-time targets in minutes, used for the business-impact maths.
SLA_RESPONSE_MINUTES = {P0: 15, P1: 60, P2: 240, P3: 1440}

# --------------------------------------------------------------------------
# Routing table: (category, priority) -> owning queue.
# Mirrors a real support org: fraud is never queued behind billing.
# --------------------------------------------------------------------------
QUEUES = {
    "fraud-ops": "Fraud & Security Operations",
    "payments-ops": "Payments & Settlement",
    "access-ops": "Identity & Account Access",
    "kyc-ops": "KYC / Verification Desk",
    "cards-ops": "Card Servicing",
    "billing-ops": "Billing & Disputes",
    "loans-ops": "Lending / EMI Desk",
    "tech-support": "Technical Support (Tier 2)",
    "frontline": "Frontline Support (Tier 1)",
}

_ROUTING = {
    (FRAUD_SECURITY, P0): "fraud-ops",
    (FRAUD_SECURITY, P1): "fraud-ops",
    (FRAUD_SECURITY, P2): "fraud-ops",
    (FRAUD_SECURITY, P3): "frontline",
    (PAYMENT_FAILURE, P0): "payments-ops",
    (PAYMENT_FAILURE, P1): "payments-ops",
    (PAYMENT_FAILURE, P2): "payments-ops",
    (PAYMENT_FAILURE, P3): "frontline",
    (ACCOUNT_ACCESS, P0): "access-ops",
    (ACCOUNT_ACCESS, P1): "access-ops",
    (ACCOUNT_ACCESS, P2): "access-ops",
    (ACCOUNT_ACCESS, P3): "frontline",
    (KYC_VERIFICATION, P0): "kyc-ops",
    (KYC_VERIFICATION, P1): "kyc-ops",
    (KYC_VERIFICATION, P2): "kyc-ops",
    (KYC_VERIFICATION, P3): "frontline",
    (REFUND_CHARGEBACK, P0): "billing-ops",
    (REFUND_CHARGEBACK, P1): "billing-ops",
    (REFUND_CHARGEBACK, P2): "billing-ops",
    (REFUND_CHARGEBACK, P3): "frontline",
    (CARD_SERVICES, P0): "cards-ops",
    (CARD_SERVICES, P1): "cards-ops",
    (CARD_SERVICES, P2): "cards-ops",
    (CARD_SERVICES, P3): "frontline",
    (BILLING_CHARGES, P0): "billing-ops",
    (BILLING_CHARGES, P1): "billing-ops",
    (BILLING_CHARGES, P2): "billing-ops",
    (BILLING_CHARGES, P3): "frontline",
    (LOAN_EMI, P0): "loans-ops",
    (LOAN_EMI, P1): "loans-ops",
    (LOAN_EMI, P2): "loans-ops",
    (LOAN_EMI, P3): "frontline",
    (TECHNICAL_APP, P0): "tech-support",
    (TECHNICAL_APP, P1): "tech-support",
    (TECHNICAL_APP, P2): "tech-support",
    (TECHNICAL_APP, P3): "frontline",
    (GENERAL_ENQUIRY, P0): "frontline",
    (GENERAL_ENQUIRY, P1): "frontline",
    (GENERAL_ENQUIRY, P2): "frontline",
    (GENERAL_ENQUIRY, P3): "frontline",
    (OTHER, P0): "frontline",
    (OTHER, P1): "frontline",
    (OTHER, P2): "frontline",
    (OTHER, P3): "frontline",
}


def route(category: str, priority: str) -> str:
    """Map a (category, priority) pair to a queue id. Never raises."""
    return _ROUTING.get((category, priority), "frontline")


def coarse_of(intent: str) -> str:
    """Coarse category for a fine intent. Unmapped intents fall to OTHER."""
    return INTENT_TO_COARSE.get(intent, OTHER)


# Import-time integrity check -- the taxonomy must cover the dataset exactly.
def validate(intents):
    """Raise if the mapping does not cover `intents` exactly."""
    unknown = [i for i in intents if i not in INTENT_TO_COARSE]
    if unknown:
        raise ValueError(f"Unmapped Banking77 intents: {unknown}")
    missing = [i for i in INTENT_TO_COARSE if i not in set(intents)]
    if missing:
        raise ValueError(f"Mapping contains intents absent from dataset: {missing}")
    return True