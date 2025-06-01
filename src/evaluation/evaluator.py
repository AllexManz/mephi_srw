"""
Main evaluator module for security language model.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from .metrics import calculate_perplexity, calculate_accuracy
from ..training.dataset import SecurityDataset

class SecurityModelEvaluator:
    """Класс для оценки модели безопасности."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: DictConfig,
        output_dir: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        else:
            self.writer = None
    
    def evaluate(
        self,
        eval_dataset: Union[SecurityDataset, List[Dict[str, str]]],
        batch_size: Optional[int] = None,
        device: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Оценка модели на заданном датасете.
        
        Args:
            eval_dataset: Датасет для оценки
            batch_size: Размер батча (если None, берется из конфига)
            device: Устройство для вычислений (если None, определяется автоматически)
            
        Returns:
            Dict[str, float]: Словарь с метриками оценки
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = batch_size or self.cfg.evaluation.batch_size
        
        # Переводим модель в режим оценки
        self.model.eval()
        self.model.to(device)
        
        metrics = {}
        
        # Вычисляем перплексию если передан SecurityDataset
        if isinstance(eval_dataset, SecurityDataset):
            dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.cfg.evaluation.num_workers
            )
            
            perplexity = calculate_perplexity(
                self.model,
                dataloader,
                device=device
            )
            metrics["perplexity"] = perplexity
            
            if self.writer:
                self.writer.add_scalar("evaluation/perplexity", perplexity)
        
        # Вычисляем точность если передан список примеров
        if isinstance(eval_dataset, list):
            accuracy_metrics = calculate_accuracy(
                self.model,
                self.tokenizer,
                eval_dataset,
                max_length=self.cfg.evaluation.max_length,
                device=device
            )
            metrics.update(accuracy_metrics)
            
            if self.writer:
                for name, value in accuracy_metrics.items():
                    self.writer.add_scalar(f"evaluation/{name}", value)
        
        # Сохраняем метрики в файл если указана директория
        if self.output_dir:
            metrics_file = self.output_dir / "evaluation_metrics.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Закрываем TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return metrics
    
    def generate_responses(
        self,
        examples: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        device: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Генерация ответов модели на заданных примерах.
        
        Args:
            examples: Список примеров для генерации
            max_new_tokens: Максимальное количество новых токенов
            temperature: Температура генерации
            device: Устройство для вычислений
            
        Returns:
            List[Dict[str, str]]: Список примеров с сгенерированными ответами
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        max_new_tokens = max_new_tokens or self.cfg.evaluation.max_new_tokens
        temperature = temperature or self.cfg.evaluation.temperature
        
        self.model.eval()
        self.model.to(device)
        
        results = []
        
        for example in examples:
            # Формируем промпт
            prompt = f"""### Инструкция: {example['instruction']}

### Контекст: {example.get('context', '')}

### Вход: {example['input']}

### Ответ:"""
            
            # Токенизируем вход
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.cfg.evaluation.max_length,
                truncation=True
            ).to(device)
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодируем ответ
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем сгенерированный ответ
            try:
                generated_answer = generated_text.split("### Ответ:")[-1].strip()
            except IndexError:
                generated_answer = ""
            
            # Сохраняем результат
            result = {
                "instruction": example["instruction"],
                "context": example.get("context", ""),
                "input": example["input"],
                "expected_output": example["output"],
                "generated_output": generated_answer
            }
            results.append(result)
        
        # Сохраняем результаты в файл если указана директория
        if self.output_dir:
            results_file = self.output_dir / "generated_responses.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results 