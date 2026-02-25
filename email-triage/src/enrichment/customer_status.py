"""
Customer status resolution — purely deterministic, CRM-first.

The LLM is NEVER the decision-maker here. It may surface evidence
(e.g. "ho già un contratto con voi") but the final value comes from
CRM lookup or text signals, in that order.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Literal, Protocol

from src.contracts import CustomerStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CRM lookup protocol
# ─────────────────────────────────────────────────────────────────────────────

class CRMLookupResult:
    __slots__ = ("match_type", "confidence")

    def __init__(
        self,
        match_type: Literal["exact", "domain", "none", "error"],
        confidence: float,
    ) -> None:
        self.match_type = match_type
        self.confidence = confidence


class CRMLookup(Protocol):
    def __call__(self, email_address: str) -> CRMLookupResult:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Text signal patterns (Italian)
# ─────────────────────────────────────────────────────────────────────────────

_EXISTING_SIGNALS: list[str] = [
    "ho già un contratto",
    "sono già vostro cliente",
    "cliente dal",
    "vostro cliente",
    "mio contratto",
    "già cliente",
    "codice cliente",
    "numero pratica",
    "pratica n.",
    "pratica nr",
]


def _has_existing_signal(text: str) -> bool:
    text_lower = text.lower()
    return any(signal in text_lower for signal in _EXISTING_SIGNALS)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_customer_status(
    from_email: str,
    crm_lookup: CRMLookup | None,
    text_body: str = "",
) -> CustomerStatus:
    """
    Resolve customer status deterministically.

    Resolution order:
    1. CRM exact match          → existing (conf=1.0)
    2. CRM domain match         → existing (conf=0.7)
    3. Text signal detected     → existing (conf=0.5)
    4. No signal                → new     (conf=0.8)
    5. CRM lookup error         → unknown (conf=0.2)
    """
    if crm_lookup is None:
        # No CRM integration: fall back to text signals only
        if _has_existing_signal(text_body):
            return CustomerStatus(value="existing", confidence=0.5, source="text_signal")
        return CustomerStatus(value="new", confidence=0.6, source="no_crm_configured")

    try:
        result = crm_lookup(from_email)
    except Exception as exc:
        logger.error("CRM lookup failed for %s: %s", from_email, exc)
        return CustomerStatus(value="unknown", confidence=0.2, source="lookup_error")

    if result.match_type == "exact":
        return CustomerStatus(
            value="existing", confidence=1.0, source="crm_exact_match"
        )
    if result.match_type == "domain":
        return CustomerStatus(
            value="existing", confidence=0.7, source="crm_domain_match"
        )
    # match_type == "none"
    if _has_existing_signal(text_body):
        return CustomerStatus(
            value="existing", confidence=0.5, source="text_signal"
        )
    return CustomerStatus(value="new", confidence=0.8, source="no_crm_no_signal")
