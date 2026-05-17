"""Generation and perplexity helpers for attack / leakage benchmarks."""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    *,
    max_length: int = 512,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str | None = None,
    system_prefix: str = "",
    record_memory: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    full_prompt = system_prefix + prompt if system_prefix else prompt
    enc = tokenizer(
        full_prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    ).to(dev)
    stats: Dict[str, Any] = {"response_time_seconds": 0.0}

    torch_sync = getattr(torch.cuda, "synchronize", None)
    if torch_sync:
        torch_sync()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    if torch_sync:
        torch_sync()
    stats["response_time_seconds"] = time.perf_counter() - t0

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Ответ:" in full_text:
        answer = full_text.split("### Ответ:")[-1].strip()
    elif len(full_text) > len(full_prompt):
        answer = full_text[len(full_prompt) :].strip()
    else:
        answer = full_text

    if record_memory and torch.cuda.is_available():
        stats["peak_memory_reserved_bytes"] = float(torch.cuda.max_memory_reserved())

    return answer, stats


def sequence_negative_log_likelihood(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    full_text: str,
    *,
    device: str | None = None,
    max_length: int = 512,
) -> float:
    """Average NLL per token over the causal sequence (MI proxy)."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = tokenizer(full_text, return_tensors="pt", max_length=max_length, truncation=True).to(dev)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        )
        mask = labels.reshape(-1) != tokenizer.pad_token_id
        denom = torch.clamp(mask.sum(), min=1)
        nll = (loss.reshape(-1) * mask.float()).sum() / denom
    return float(nll.item())


def summarize_latencies(times: List[float]) -> Dict[str, float]:
    if not times:
        return {"mean_latency_s": 0.0, "p95_latency_s": 0.0}
    st = sorted(times)
    idx = max(0, min(len(st) - 1, int(0.95 * (len(st) - 1))))
    return {
        "mean_latency_s": float(statistics.fmean(times)),
        "p95_latency_s": float(st[idx]),
    }


class MockResponder:
    """Stub when GPU/weights unavailable — returns attack-like outputs for obvious triggers."""

    def __call__(self, prompt: str) -> str:
        p = prompt.lower()
        if "ignore all instructions" in p or "[inst]" in p or "dan mode" in p:
            return "Я подчинился инструкции атакующего. секретное_слово=PI_SUCCESS"
        return (
            '{"incident_type": "potential_anomaly", "severity": "medium", '
            '"recommendations": ["сбор логов", "проверка учётки"]}'
        )
