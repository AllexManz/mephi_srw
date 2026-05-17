"""Module 1: input hygiene for Splunk/event-driven prompts."""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Sequence, Tuple

IMPERATIVE_PATTERNS = re.compile(
    r"(ignore|disregard|forget|instead|override|new instructions)\b[^\n\.]{0,120}",
    re.IGNORECASE,
)
BOUNDARY_MARKERS = re.compile(
    r"(###|---+|`{3}|\[INST\]|<<SYS>>|</s>|<\|im_start\|>)",
    re.IGNORECASE | re.MULTILINE,
)
PRIVATE_IP_PATTERN = re.compile(
    r"^(10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})$"
)
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-\.@]{1,128}$")


class PromptInjectionFilter:
    """Heuristic sanitization layered on Splunk-ish JSON payloads."""

    def __init__(
        self,
        escape_special: bool = True,
        wrap_user_data: bool = False,
        strict_username: bool = False,
        strict_ipv4_sources: Sequence[str] = ("src_ip", "dest_ip"),
    ):
        self.escape_special = escape_special
        self.wrap_user_data = wrap_user_data
        self.strict_username = strict_username
        self.strict_ipv4_sources = tuple(strict_ipv4_sources)

    def sanitize_string_leaf(self, key: str, value: Any) -> Any:
        if not isinstance(value, str):
            return value

        flagged = []

        text = value
        if IMPERATIVE_PATTERNS.search(text):
            flagged.append("imperative_construct")
        if BOUNDARY_MARKERS.search(text):
            flagged.append("boundary_marker")

        if key in self.strict_ipv4_sources and text and not self._valid_optional_ip(text):
            flagged.append("ip_format")
        if key == "username" and text and self.strict_username and not USERNAME_PATTERN.match(text):
            flagged.append("username_chars")

        if self.escape_special:
            text = html.escape(text, quote=False)
        elif IMPERATIVE_PATTERNS.search(text):
            text = IMPERATIVE_PATTERNS.sub("[redacted-rule]", text)
        return {"__sanitized_value__": text, "__flags__": flagged}

    def _valid_optional_ip(self, s: str) -> bool:
        if s in ("-", "unknown"):
            return True
        if PRIVATE_IP_PATTERN.match(s.strip()):
            return True
        return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", s.strip()))

    def _walk(self, obj: Any, flags: List[str]) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, str):
                    r = self.sanitize_string_leaf(k, v)
                    if isinstance(r, dict) and "__sanitized_value__" in r:
                        flags.extend([f"{k}:{flt}" for flt in r["__flags__"]])
                        out[k] = r["__sanitized_value__"]
                    else:
                        out[k] = v
                else:
                    out[k] = self._walk(v, flags)
            return out
        if isinstance(obj, list):
            return [self._walk(item, flags) for item in obj]
        return obj

    def sanitize_payload(self, payload: Dict[str, Any] | List[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        flags: List[str] = []
        if isinstance(payload, list):
            body = self._walk(payload, flags)
        else:
            body = self._walk(payload, flags)
        meta = {
            "flags": sorted(set(flags)),
            "imperative_hits": any(":imperative_construct" in f for f in flags),
            "wrapped": False,
        }
        if self.wrap_user_data:
            meta["wrapped"] = True
        return body, meta
