"""
Metrics calculation for model evaluation.
"""

import math
import re
from typing import Dict, List, Union, Optional, Any
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

def calculate_perplexity(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    """
    Вычисляет перплексию модели на заданном датасете.

    Args:
        model: Модель для оценки
        dataloader: DataLoader с данными для оценки
        device: Устройство для вычислений

    Returns:
        float: Значение перплексии
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity"):
            # Переносим батч на нужное устройство
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Получаем loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Считаем количество токенов (исключая padding)
            num_tokens = attention_mask.sum().item()
            total_tokens += num_tokens
            total_loss += outputs.loss.item() * num_tokens

    # Пустой eval-набор возможен, например, при слишком маленьком датасете
    # и split с долей 1.0. В этом случае метрика неопределена, но оценка не
    # должна падать с ZeroDivisionError.
    if total_tokens == 0:
        return float("nan")

    # Вычисляем средний loss и перплексию
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity

def calculate_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: List[Dict[str, str]],
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """
    Вычисляет точность модели на заданных примерах.

    Args:
        model: Модель для оценки
        tokenizer: Токенизатор
        examples: Список примеров для оценки
        max_length: Максимальная длина последовательности
        device: Устройство для вычислений

    Returns:
        Dict[str, float]: Словарь с метриками точности
    """
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    for example in tqdm(examples, desc="Calculating accuracy"):
        # Формируем промпт
        prompt = f"""### Инструкция: {example['instruction']}

### Контекст: {example.get('context', '')}

### Вход: {example['input']}

### Ответ:"""

        # Токенизируем вход
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(device)

        # Генерируем ответ
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Декодируем ответ
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлекаем сгенерированный ответ (после "### Ответ:")
        try:
            generated_answer = generated_text.split("### Ответ:")[-1].strip()
        except IndexError:
            generated_answer = ""

        # Сравниваем с правильным ответом
        if generated_answer.strip() == example["output"].strip():
            correct_predictions += 1
        total_predictions += 1

    # Вычисляем метрики
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return {
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions
    }


def _parse_threat_level(text: str) -> Optional[str]:
    """Parse threat level from text in RU/EN."""
    if not text:
        return None
    lowered = text.lower()
    if "высок" in lowered or "high" in lowered:
        return "high"
    if "средн" in lowered or "medium" in lowered:
        return "medium"
    if "низк" in lowered or "low" in lowered:
        return "low"
    return None


def _extract_mitre_ids(text: str) -> List[str]:
    """Extract MITRE ATT&CK IDs like T1059 or T1059.001."""
    if not text:
        return []
    return re.findall(r"\bT\d{4}(?:\.\d{3})?\b", text.upper())


def _extract_numeric(text: str) -> Optional[float]:
    """Extract first float from text."""
    if not text:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def calculate_security_metrics(
    examples: List[Dict[str, Any]],
    responses: List[Dict[str, Any]]
) -> Dict[str, Optional[float]]:
    """
    Calculate security-specific metrics from evaluation examples and model responses.

    Expected optional fields in examples:
      - threat_level (low/medium/high or RU)
      - is_threat (bool)
      - reward (float)
      - expert_preferred_output (str)
      - mitre_attack (List[str])  # MITRE IDs like T1059
      - sequence_id (str/int)     # for sequential handling
      - sequence_index (int)      # order in sequence
      - attack_stages (List[str]) # stage labels
      - temporal_label (str)      # temporal context label
    """
    if not examples or not responses:
        return {
            "average_reward": None,
            "threat_detection_accuracy": None,
            "false_positive_rate": None,
            "false_negative_rate": None,
            "expert_preference_rate": None,
            "mitre_attack_coverage": None,
            "sequential_event_handling": None,
            "average_response_time": None,
            "multi_stage_attack_detection": None,
            "temporal_context_accuracy": None
        }

    # Index responses by (instruction,input) fallback by position
    if len(responses) == len(examples):
        paired = list(zip(examples, responses))
    else:
        resp_by_key = {
            (r.get("instruction"), r.get("input")): r for r in responses
        }
        paired = []
        for ex in examples:
            resp = resp_by_key.get((ex.get("instruction"), ex.get("input")))
            if resp:
                paired.append((ex, resp))

    rewards = []
    total = 0
    correct_threat = 0
    false_pos = 0
    false_neg = 0
    positives = 0
    negatives = 0
    expert_total = 0
    expert_match = 0
    mitre_total = 0
    mitre_hit = 0
    sequence_total = 0
    sequence_ok = 0
    response_times = []
    stage_total = 0
    stage_hit = 0
    temporal_total = 0
    temporal_hit = 0

    for ex, resp in paired:
        total += 1
        output = resp.get("generated_output", "")

        # Average Reward
        reward_val = ex.get("reward")
        if reward_val is None:
            reward_val = _extract_numeric(resp.get("reward", ""))
            if reward_val is None:
                reward_val = _extract_numeric(output)
        if reward_val is not None:
            rewards.append(float(reward_val))

        # Threat metrics
        gt_threat = ex.get("threat_level")
        if gt_threat is None and "is_threat" in ex:
            gt_threat = "high" if ex.get("is_threat") else "low"
        pred_threat = _parse_threat_level(output)
        if gt_threat is not None and pred_threat is not None:
            gt_norm = _parse_threat_level(str(gt_threat)) or str(gt_threat).lower()
            is_gt_pos = gt_norm in {"high", "medium"}
            is_pred_pos = pred_threat in {"high", "medium"}
            positives += int(is_gt_pos)
            negatives += int(not is_gt_pos)
            correct_threat += int(is_gt_pos == is_pred_pos)
            false_pos += int((not is_gt_pos) and is_pred_pos)
            false_neg += int(is_gt_pos and (not is_pred_pos))

        # Expert preference
        expert_output = ex.get("expert_preferred_output")
        if expert_output:
            expert_total += 1
            if output.strip() == expert_output.strip():
                expert_match += 1

        # MITRE coverage
        expected_mitre = ex.get("mitre_attack")
        if expected_mitre:
            expected_ids = {item.upper() for item in expected_mitre}
            found_ids = set(_extract_mitre_ids(output))
            mitre_total += len(expected_ids)
            mitre_hit += len(expected_ids & found_ids)

        # Sequential handling
        if "sequence_id" in ex and "sequence_index" in ex:
            sequence_total += 1
            # Check that output mentions sequence order index
            expected_idx = str(ex.get("sequence_index"))
            if expected_idx in output:
                sequence_ok += 1

        # Average response time
        if "response_time_seconds" in resp:
            response_times.append(float(resp["response_time_seconds"]))

        # Multi-stage attack detection
        stages = ex.get("attack_stages")
        if stages:
            stage_total += 1
            found = 0
            for stage in stages:
                if stage.lower() in output.lower():
                    found += 1
            if found > 0:
                stage_hit += 1

        # Temporal context accuracy
        temporal_label = ex.get("temporal_label")
        if temporal_label:
            temporal_total += 1
            if str(temporal_label).lower() in output.lower():
                temporal_hit += 1

    metrics = {
        "average_reward": sum(rewards) / len(rewards) if rewards else None,
        "threat_detection_accuracy": (correct_threat / (positives + negatives)) if (positives + negatives) else None,
        "false_positive_rate": (false_pos / negatives) if negatives else None,
        "false_negative_rate": (false_neg / positives) if positives else None,
        "expert_preference_rate": (expert_match / expert_total) if expert_total else None,
        "mitre_attack_coverage": (mitre_hit / mitre_total) if mitre_total else None,
        "sequential_event_handling": (sequence_ok / sequence_total) if sequence_total else None,
        "average_response_time": (sum(response_times) / len(response_times)) if response_times else None,
        "multi_stage_attack_detection": (stage_hit / stage_total) if stage_total else None,
        "temporal_context_accuracy": (temporal_hit / temporal_total) if temporal_total else None
    }

    return metrics
