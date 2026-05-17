"""Module 4: redact plausible secrets from generations."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple

PRIVATE_IP_PATTERN = re.compile(
    r"\b(10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})\b"
)

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
GENERIC_SECRET_PATTERN = re.compile(
    r"\b(sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9\-_]{15,}|"
    r"api[_\-]?key['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9\-_]{16,})", re.I
)


def redact_sensitive(text: str) -> Tuple[str, Dict[str, int]]:
    """Return redacted copy and coarse hit counts."""

    counters: Dict[str, int] = {}
    replaced = PRIVATE_IP_PATTERN.subn("[REDACTED_PRIV_IP]", text)
    text, counters["private_ips"] = replaced
    replaced = EMAIL_PATTERN.subn("[REDACTED_EMAIL]", text)
    text, counters["emails"] = replaced
    replaced = JWT_PATTERN.subn("[REDACTED_JWT]", text)
    text, counters["jwt"] = replaced
    replaced = GENERIC_SECRET_PATTERN.subn("[REDACTED_CREDENTIAL]", text)
    text, counters["generic_secrets"] = replaced
    return text, counters


def ngrams(text: str, n: int) -> Iterable[str]:
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    for i in range(0, max(0, len(toks) - n + 1)):
        yield " ".join(toks[i : i + n])


def cve_quotation_overlap(response: str, reference_snippets: Iterable[str]) -> float:
    """Share of trigrams overlapping any reference excerpt (cheap verbatim-leak heuristic)."""

    rg = list(ngrams(response, 3))
    if not rg:
        return 0.0
    refs = []
    for s in reference_snippets:
        refs.extend(ngrams(s, 3))
    ref_set = set(refs)
    if not ref_set:
        return 0.0
    hits = sum(1 for g in rg if g in ref_set)
    return hits / len(rg)
