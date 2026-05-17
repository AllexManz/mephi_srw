"""Convert Splunk-like JSON events to the same instruction-style prompt used in SIEM/EventProcessor."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Union

SplunkPayload = Union[Dict[str, Any], List[Dict[str, Any]]]


def normalize_events(payload: SplunkPayload) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    return [payload]


def events_json_block(events: Sequence[Dict[str, Any]]) -> str:
    blocks = []
    for i, event in enumerate(events):
        blocks.append(f"Event {i + 1}:\n{json.dumps(event, ensure_ascii=False, indent=2)}")
    return "\n".join(blocks)


def splunk_payload_to_instruction_format(
    payload: SplunkPayload,
    instruction: str | None = None,
    *,
    wrap_user_data: bool = False,
) -> Dict[str, str]:
    """Return instruction-style fields compatible with evaluator prompt construction."""
    events = normalize_events(payload)
    body = events_json_block(events)
    if wrap_user_data:
        body = f"<USER_DATA>\n{body}\n</USER_DATA>"

    inst = instruction or (
        "Проанализируй следующие события Splunk/SIEM. Опиши возможный инцидент, серьёзность "
        "(низкая/средняя/высокая) и дай краткие рекомендации в форме структурированного ответа."
    )
    return {
        "instruction": inst,
        "context": "",
        "input": body,
    }


def evaluator_style_prompt(example: Dict[str, str]) -> str:
    """Mirror SecurityModelEvaluator / train.py textual format."""
    return (
        f"### Инструкция: {example['instruction']}\n\n"
        f"### Контекст: {example.get('context', '')}\n\n"
        f"### Вход: {example['input']}\n\n"
        f"### Ответ:"
    )
