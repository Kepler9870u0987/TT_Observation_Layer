"""
Email preprocessing: quote/signature/disclaimer stripping + building EmailDocument.

Design constraints:
- All regex patterns are VERSIONED (CANONICALIZATION_VERSION).
- Removed sections are logged for audit (not discarded silently).
- Same input → same output (deterministic, no RNG).
"""
from __future__ import annotations

import re
from datetime import datetime

from src.contracts import EmailDocument, RemovedSection
from src.ingestion.parser import extract_body_text, get_headers, parse_eml_bytes

# ─────────────────────────────────────────────────────────────────────────────
# Version constants (bump when patterns change)
# ─────────────────────────────────────────────────────────────────────────────

PARSER_VERSION = "email-parser-1.3.0"
CANONICALIZATION_VERSION = "1.2.0"

# ─────────────────────────────────────────────────────────────────────────────
# Quote / signature / disclaimer patterns
# Order matters: patterns are applied top-to-bottom.
# ─────────────────────────────────────────────────────────────────────────────

_STRIP_PATTERNS: list[tuple[str, str]] = [
    # Reply headers (Italian + English)
    (r"(?is)\nIl\s+giorno\s+.{1,100}\s+ha\s+scritto:.*$", "reply_header"),
    (r"(?is)\nOn\s+.{1,100}\s+wrote:.*$", "reply_header"),
    (r"(?is)\n-{5,}\s*Messaggio\s+originale\s*-{5,}.*$", "reply_header"),
    (r"(?is)\n-{5,}\s*Original\s+message\s*-{5,}.*$", "reply_header"),
    # Quoted lines (> prefix)
    (r"(?m)^[ \t]*>.*$", "quote"),
    # Signatures (double dash or underscore block)
    (r"(?m)^--\s*$[\s\S]*", "signature"),
    (r"(?m)^_{10,}[\s\S]*$", "signature"),
    # Common disclaimers
    (
        r"(?is)(questo\s+messaggio|this\s+(e-?mail|message)).{0,50}"
        r"(riservato|confidential|privileged).*$",
        "disclaimer",
    ),
]

# Salutations that can safely be removed from the END of the body
_SIGNATURE_SALUTATIONS = re.compile(
    r"(?im)^(cordiali\s+saluti|distinti\s+saluti|saluti|"
    r"con\s+ossequi|in\s+fede|regards|best\s+regards|"
    r"kind\s+regards|sinceramente)[,.]?\s*$"
    r"([\s\S]{0,200})?$"
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def canonicalize_text(
    text: str,
    keep_audit_info: bool = True,
) -> tuple[str, list[RemovedSection]]:
    """
    Strip quotes, signatures, and disclaimers from email body.

    Returns:
        (cleaned_text, list[RemovedSection])

    The function is purely deterministic: same input → same output.
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    removed: list[RemovedSection] = []

    for pattern, section_type in _STRIP_PATTERNS:
        for match in re.finditer(pattern, text):
            if keep_audit_info:
                removed.append(
                    RemovedSection(
                        section_type=section_type,
                        span_start=match.start(),
                        span_end=match.end(),
                        content=match.group(0)[:500],  # cap content for storage
                    )
                )
        text = re.sub(pattern, "\n", text)

    # Remove closing salutation block
    sal_match = _SIGNATURE_SALUTATIONS.search(text)
    if sal_match:
        if keep_audit_info:
            removed.append(
                RemovedSection(
                    section_type="signature",
                    span_start=sal_match.start(),
                    span_end=sal_match.end(),
                    content=sal_match.group(0)[:200],
                )
            )
        text = text[: sal_match.start()]

    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), removed


def build_email_document(
    raw: bytes,
    parser_version: str = PARSER_VERSION,
    canonicalization_version: str = CANONICALIZATION_VERSION,
) -> EmailDocument:
    """
    Full pipeline: bytes → EmailDocument with canonical body.

    Raises:
        ValueError: if Message-ID is missing (required for idempotency).
    """
    from email.message import EmailMessage

    msg: EmailMessage = parse_eml_bytes(raw)
    headers = get_headers(msg)

    message_id = headers["message_id"].strip("<>").strip() or ""
    if not message_id:
        # Fallback: derive a stable id from content hash
        import hashlib
        body_raw = extract_body_text(msg)
        message_id = "derived-" + hashlib.sha1(
            (headers["from"] + headers["date"] + body_raw[:200]).encode()
        ).hexdigest()

    body = extract_body_text(msg)
    body_canonical, removed = canonicalize_text(body, keep_audit_info=True)

    return EmailDocument(
        message_id=message_id,
        from_raw=headers["from"],
        to_raw=headers["to"],
        subject=headers["subject"],
        body=body,
        body_canonical=body_canonical,
        removed_sections=[
            {
                "type": r.section_type,
                "span_start": r.span_start,
                "span_end": r.span_end,
                "content": r.content,
            }
            for r in removed
        ],
        parser_version=parser_version,
        canonicalization_version=canonicalization_version,
        received_at=datetime.utcnow(),
    )
