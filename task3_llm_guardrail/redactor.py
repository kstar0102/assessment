"""
Streaming PII redactor.

The core problem: PII can be split across chunks.
    chunk 1: "mail me at jo"
    chunk 2: "hn@acme.io or call"
If we redact chunk by chunk we leak "jo" and miss the rest.

Approach: a small holdback buffer.
  * Append each delta to the buffer.
  * Find the "unsafe tail": the shortest suffix that could still grow into
    a match (a trailing word that might be an email, or a trailing run of
    digits and separators that might be a card or SSN).
  * Everything before the tail is final. Redact it and emit it now.
  * Keep only the tail. It's at most a word or a number long, so memory is
    O(longest PII token), never O(response).
  * A hard cap (MAX_HOLDBACK) forces a flush if someone streams a giant
    unbroken token, so latency stays bounded too.

Patterns
  email  RFC ish local@domain.tld
  ssn    123-45-6789 / 123 45 6789 / 123456789, with SSA invalid ranges excluded
  card   13 to 19 digits with optional space/dash separators, Luhn checked so
         order numbers and timestamps aren't nuked
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"
MAX_HOLDBACK = 256

_EMAIL = r"(?P<email>[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,})"
_CARD = r"(?P<card>(?<![\d])(?:\d[ \-]?){12,18}\d(?![\d]))"
_SSN = r"(?P<ssn>(?<![\d\-])(?!000|666|9\d\d)\d{3}(?P<sep>[\- ]?)(?!00)\d{2}(?P=sep)(?!0000)\d{4}(?![\d]))"

PII_RE = re.compile("|".join([_EMAIL, _CARD, _SSN]))

# Suffixes that might still become PII once more text arrives
_TAIL_WORD = re.compile(r"[A-Za-z0-9._%+\-@]+$")          # possible email
_TAIL_DIGITS = re.compile(r"(?:\d[ \-]?)+$")               # possible card/ssn


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _replace(m: re.Match) -> str:
    if m.group("card") is not None:
        digits = re.sub(r"\D", "", m.group("card"))
        if not luhn_ok(digits):
            return m.group(0)
    return REDACTED


def redact(text: str) -> str:
    return PII_RE.sub(_replace, text)


class StreamRedactor:
    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = ""

    def _safe_split(self) -> int:
        buf = self._buf
        cut = len(buf)
        for rx in (_TAIL_WORD, _TAIL_DIGITS):
            m = rx.search(buf)
            if m:
                cut = min(cut, m.start())
        # Never split through the middle of a match that's already complete
        for m in PII_RE.finditer(buf, max(0, cut - MAX_HOLDBACK)):
            if m.start() < cut < m.end():
                cut = m.start()
        return cut

    def feed(self, delta: str) -> str:
        """Add a delta, return whatever text is now safe to emit (maybe '')."""
        if not delta:
            return ""
        self._buf += delta
        cut = self._safe_split()
        if cut == 0 and len(self._buf) > MAX_HOLDBACK:
            cut = len(self._buf)  # bounded latency over a pathological token
        if cut == 0:
            return ""
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return redact(out)

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return redact(out) if out else ""

    @property
    def pending(self) -> int:
        return len(self._buf)
