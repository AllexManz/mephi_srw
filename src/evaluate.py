"""
Main script for model evaluation.
"""

import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.trainer import SecurityModelTrainer
from evaluation.evaluator import SecurityModelEvaluator
from training.dataset import SecurityDataset

@hydra.main(version_base=None, config_path="../configs", config_name="evaluation/default")
def main(cfg: DictConfig) -> None:
    """Основная функция для оценки модели."""
    # Создаем директорию для результатов
    output_dir = Path(cfg.evaluation.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем модель и токенизатор
    print(f"Loading model from {cfg.model.model.name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model.name,
        torch_dtype=getattr(torch, cfg.model.model.torch_dtype),
        device_map=cfg.model.model.device_map,
        trust_remote_code=cfg.model.model.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model.name)
    
    # Если используется PEFT, загружаем адаптер
    if cfg.peft.enabled:
        print(f"Loading PEFT adapter from {cfg.peft.adapter_path}...")
        trainer = SecurityModelTrainer.load_model(cfg.peft.adapter_path, cfg)
        model = trainer.model
        tokenizer = trainer.tokenizer
    
    # Загружаем датасет для оценки
    print(f"Loading evaluation dataset from {cfg.data.eval_path}...")
    with open(cfg.data.eval_path, "r", encoding="utf-8") as f:
        eval_examples = json.load(f)
    
    # Создаем датасет
    eval_dataset = SecurityDataset(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_length=cfg.evaluation.max_length
    )
    
    # Инициализируем оценщик
    evaluator = SecurityModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        output_dir=str(output_dir)
    )
    
    # Запускаем оценку
    print("Starting evaluation...")
    metrics = evaluator.evaluate(eval_dataset)
    
    # Выводим результаты
    print("\nEvaluation results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Генерируем ответы на примерах
    print("\nGenerating responses...")
    results = evaluator.generate_responses(eval_examples)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main() 