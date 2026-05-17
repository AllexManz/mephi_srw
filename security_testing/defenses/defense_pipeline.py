"""Assemble sanitization → generation → validation layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from security_testing.common.repo_paths import find_repo_root
from security_testing.defenses.input_sanitizer import PromptInjectionFilter
from security_testing.defenses.output_validator import OutputValidator
from security_testing.defenses.sensitive_data_filter import redact_sensitive
from security_testing.defenses.system_prompt import hardened_system_prefix
from security_testing.prompt_conversion import evaluator_style_prompt, splunk_payload_to_instruction_format


class DefensePipeline:
    def __init__(
        self,
        defense_yaml: Path | None = None,
        enable_input: bool | None = None,
        wrap_user_data_tag: bool = True,
        redact_output: bool = True,
        require_strict_json_output: bool = False,
        canary: str | None = None,
        topical_en: list[str] | None = None,
        topical_ru: list[str] | None = None,
    ):
        repo = find_repo_root()
        dpath = defense_yaml or repo / "security_testing" / "configs" / "defense_config.yaml"
        cfg: Dict[str, Any] = {}
        if isinstance(dpath, Path) and dpath.is_file():
            with open(dpath, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        self.canary = canary or cfg.get("prompt_canary_token")

        iff = cfg.get("input_filter", {})
        inp_enabled = enable_input if enable_input is not None else iff.get("enabled", True)

        esc = iff.get("escape_special", True)
        self.prompt_filter = PromptInjectionFilter(
            escape_special=esc,
            wrap_user_data=False,
            strict_username=False,
        )

        self.wrap_user_data_tag = wrap_user_data_tag
        self.enabled_input_filter = inp_enabled

        oval = cfg.get("output_validator", {})
        require_json_attempt = oval.get(
            "require_json_attempt",
            False,
        ) or require_strict_json_output

        topical_en_cfg = topical_en if topical_en is not None else oval.get("topical_keywords_en") or []
        topical_ru_cfg = topical_ru if topical_ru is not None else oval.get("topical_keywords_ru") or []

        self.validator = OutputValidator(
            max_chars=oval.get("max_chars", 8000),
            min_chars=oval.get("min_chars", 1),
            topical_keywords_en=topical_en_cfg if topical_en_cfg else None,
            topical_keywords_ru=topical_ru_cfg if topical_ru_cfg else None,
            require_json_attempt=require_json_attempt,
        )

        self.redact_output_enabled = cfg.get("output_redaction", {}).get(
            "enabled", True
        ) and redact_output

        base_prefix = hardened_system_prefix(self.canary)
        self.system_prefix = base_prefix

    def prepare_prompt_from_attack_case(self, splunk_blob: Dict[str, Any] | list) -> tuple[str, Dict[str, Any]]:
        meta_input: Dict[str, Any] = {}
        filtered_payload = splunk_blob
        if self.enabled_input_filter:
            filtered_payload, meta_input = self.prompt_filter.sanitize_payload(splunk_blob)

        tmpl = splunk_payload_to_instruction_format(filtered_payload, wrap_user_data=False)
        if self.wrap_user_data_tag:
            wrapped = tmpl["input"]
            tmpl["instruction"] += "\n\nДанные пользователя ограничены тегами <USER_DATA>...</USER_DATA>."
            tmpl["input"] = f"<USER_DATA>\n{wrapped}\n</USER_DATA>"
            meta_input["wrapped_user_data_tags"] = True

        return evaluator_style_prompt(tmpl), meta_input

    def postprocess_generation(self, text: str) -> tuple[str, Dict[str, Any]]:
        md: Dict[str, Any] = {}
        cleaned = text
        if self.redact_output_enabled:
            cleaned, hit = redact_sensitive(cleaned)
            md["redaction_counters"] = hit
        vd = self.validator.validate(cleaned)
        md["output_validation"] = vd
        return cleaned, md
