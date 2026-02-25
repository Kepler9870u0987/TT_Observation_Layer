"""
Email parser: .eml bytes / file → raw headers + body text.

Uses stdlib `email` with `policy.default` for proper RFC 6532 support.
Produces a plain-text body by preferring text/plain over text/html.
"""
from __future__ import annotations

import pathlib
import re
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_eml_bytes(raw: bytes) -> EmailMessage:
    """Parse raw .eml bytes into an EmailMessage object."""
    return BytesParser(policy=policy.default).parsebytes(raw)


def parse_eml_file(path: str | pathlib.Path) -> EmailMessage:
    """Parse a .eml file from disk."""
    with open(path, "rb") as fh:
        return parse_eml_bytes(fh.read())


def get_headers(msg: EmailMessage) -> dict[str, str]:
    """Extract standard headers as a flat dict."""
    return {
        "message_id": msg.get("Message-ID", "") or "",
        "date": msg.get("Date", "") or "",
        "from": msg.get("From", "") or "",
        "to": msg.get("To", "") or "",
        "subject": msg.get("Subject", "") or "",
        "reply_to": msg.get("Reply-To", "") or "",
        "content_type": msg.get_content_type() or "",
    }


def extract_body_text(msg: EmailMessage) -> str:
    """
    Extract the best available plain-text body.
    Preference order: text/plain > text/html (stripped) > empty string.
    """
    parts = list(msg.walk()) if msg.is_multipart() else [msg]

    plain_parts: list[str] = []
    html_parts: list[str] = []

    for part in parts:
        ctype = (part.get_content_type() or "").lower()
        # Skip attachments
        if part.get_content_disposition() == "attachment":
            continue
        try:
            content = part.get_content()
        except Exception:
            continue
        if not isinstance(content, str):
            continue
        if ctype == "text/plain":
            plain_parts.append(content)
        elif ctype == "text/html":
            html_parts.append(content)

    if plain_parts:
        return "\n".join(plain_parts).strip()
    if html_parts:
        return _html_to_text("\n".join(html_parts))
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    """Minimal but safe HTML → plain-text conversion."""
    # Remove <script> and <style> blocks entirely
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    # Convert block-level tags to newlines
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p\s*>", "\n", html)
    html = re.sub(r"(?i)</(div|tr|li|h[1-6])>", "\n", html)
    # Strip remaining tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace (but preserve newlines)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in html.splitlines()]
    return "\n".join(line for line in lines if line)
