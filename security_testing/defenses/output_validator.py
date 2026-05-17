"""Module 2 + lightweight semantic topical check."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

SECURITY_HINT = re.compile(r"(incident|attack|splunk|threat|malware|\bIOC\b|\bCVE[-\s]?\d)", re.I)
RU_HINT = re.compile("(инцидент|угроз|безопас|атак|уязвим|малвар|cve)", re.I)


class OutputValidator:
    def __init__(
        self,
        max_chars: int = 8000,
        min_chars: int = 1,
        topical_keywords_en: Optional[List[str]] = None,
        topical_keywords_ru: Optional[List[str]] = None,
        require_json_attempt: bool = False,
        expected_incident_schema: Optional[Dict[str, Any]] = None,
    ):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.topical_keywords_en = topical_keywords_en or ["security"]
        self.topical_keywords_ru = topical_keywords_ru or []
        self.require_json_attempt = require_json_attempt
        self.expected_incident_schema = expected_incident_schema or {
            "incident_type": str,
            "severity": str,
            "recommendations": list,
        }

    def _topical_lexical(self, text: str) -> bool:
        low = text.lower()
        if SECURITY_HINT.search(text) or RU_HINT.search(low):
            return True
        for w in self.topical_keywords_en:
            if w.lower() in low:
                return True
        for w in self.topical_keywords_ru:
            if w.lower() in low:
                return True
        return False

    def structural_checks(self, text: str) -> List[str]:
        issues: List[str] = []
        if len(text) < self.min_chars:
            issues.append("too_short")
        if len(text) > self.max_chars:
            issues.append("too_long")
        if "\x00" in text:
            issues.append("null_byte")
        return issues

    def json_schema_relaxed(self, text: str) -> List[str]:
        issues: List[str] = []
        blob = text.strip()
        brace = blob.find("{")
        if brace == -1:
            issues.append("no_json_like_block")
            return issues
        end = blob.rfind("}")
        snippet = blob[brace : end + 1] if end > brace else blob[brace:]
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            issues.append("invalid_json_fragment")
            return issues
        for key, typ in self.expected_incident_schema.items():
            if key not in data:
                issues.append(f"missing:{key}")
            elif typ is list and not isinstance(data[key], list):
                issues.append(f"bad_type:{key}")
            elif typ is str and not isinstance(data[key], str):
                issues.append(f"bad_type:{key}")
        return issues

    def validate(self, text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "accepted": False,
            "structural_issues": [],
            "json_issues": [],
            "semantic_on_topic": False,
        }
        out["structural_issues"] = self.structural_checks(text)
        out["semantic_on_topic"] = self._topical_lexical(text)
        if self.require_json_attempt:
            out["json_issues"] = self.json_schema_relaxed(text)
        out["accepted"] = (
            not out["structural_issues"]
            and out["semantic_on_topic"]
            and (not self.require_json_attempt or not out["json_issues"])
        )
        return out
