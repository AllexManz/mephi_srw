"""
Main evaluator module for security language model.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import calculate_perplexity, calculate_accuracy, calculate_security_metrics
from training.dataset import SecurityDataset

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
        eval_cfg = self.cfg.evaluation.evaluation if "evaluation" in self.cfg.evaluation else self.cfg.evaluation
        batch_cfg = eval_cfg.batch if "batch" in eval_cfg else eval_cfg
        batch_size = batch_size or batch_cfg.size

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
                num_workers=batch_cfg.num_workers
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
        eval_cfg = self.cfg.evaluation.evaluation if "evaluation" in self.cfg.evaluation else self.cfg.evaluation
        gen_cfg = eval_cfg.generation if "generation" in eval_cfg else eval_cfg

        max_new_tokens = max_new_tokens or gen_cfg.get("max_new_tokens", 256)
        temperature = temperature or eval_cfg.get("temperature", 0.7)
        do_sample = gen_cfg.get("do_sample", True)

        # Формируем аргументы генерации
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": gen_cfg.get("num_return_sequences", 1),
            "do_sample": do_sample,
        }

        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = eval_cfg.get("top_p", 0.9)
            gen_kwargs["top_k"] = eval_cfg.get("top_k", 50)
        else:
            num_beams = eval_cfg.get("num_beams", 1)
            if num_beams > 1:
                gen_kwargs["num_beams"] = num_beams

        repetition_penalty = eval_cfg.get("repetition_penalty")
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        self.model.eval()
        self.model.to(device)

        results = []

        for example in tqdm(examples, desc="Generating responses"):
            # Формируем промпт
            prompt = f"""### Инструкция: {example['instruction']}

### Контекст: {example.get('context', '')}

### Вход: {example['input']}

### Ответ:"""

            # Токенизируем вход
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=eval_cfg.max_length,
                truncation=True
            ).to(device)

            # Генерируем ответ
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
            response_time = time.perf_counter() - start_time

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
                "generated_output": generated_answer,
                "response_time_seconds": response_time
            }
            results.append(result)

        # Сохраняем результаты в файл если указана директория
        if self.output_dir:
            results_file = self.output_dir / "generated_responses.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    def evaluate_security_metrics(
        self,
        examples: List[Dict[str, str]],
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Optional[float]]:
        """Calculate and log security-specific metrics."""
        metrics = calculate_security_metrics(examples, responses)
        if self.writer:
            for name, value in metrics.items():
                if value is not None:
                    self.writer.add_scalar(f"evaluation/{name}", value)
        return metrics

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "evaluation_metrics.json") -> None:
        """Save metrics to the evaluation output directory."""
        if not self.output_dir:
            return
        metrics_file = self.output_dir / filename
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def close(self) -> None:
        """Close any open writers."""
        if self.writer:
            self.writer.close()
