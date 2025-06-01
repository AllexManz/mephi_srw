"""
Metrics calculation for model evaluation.
"""

import math
from typing import Dict, List, Union
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