"""
Script for running model evaluation using the existing evaluator module.
"""

import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.evaluator import SecurityModelEvaluator
from training.dataset import SecurityDataset

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    # Определение путей
    if cfg.peft.peft.enabled:
        model_path = cfg.peft.peft.adapter_path or os.path.join(
            cfg.paths.adapters_dir,
            f"{cfg.peft.peft.method}_adapter"
        )
    else:
        model_path = os.path.join(cfg.paths.checkpoints_dir, "full_model")
    
    tokenizer_path = cfg.paths.tokenizer_dir
    eval_dataset_path = os.path.join(cfg.paths.dataset_dir, "eval.json")
    
    # Проверка существования файлов
    if cfg.peft.peft.enabled and not os.path.exists(model_path):
        raise FileNotFoundError(f"PEFT adapter not found at {model_path}")
    if not os.path.exists(eval_dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_dataset_path}")
    
    # Загрузка токенизатора
    tokenizer_source = tokenizer_path if os.path.exists(tokenizer_path) else cfg.model.model.name
    print(f"Loading tokenizer from {tokenizer_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Загрузка модели
    print(f"Loading model from {model_path}...")
    if cfg.peft.peft.enabled:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model.name,
            torch_dtype=getattr(torch, cfg.model.model.torch_dtype),
            device_map=cfg.model.model.device_map,
            trust_remote_code=cfg.model.model.trust_remote_code
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        if not Path(model_path).exists():
            model_path = cfg.model.model.name
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, cfg.model.model.torch_dtype),
            device_map=cfg.model.model.device_map,
            trust_remote_code=cfg.model.model.trust_remote_code
        )
    
    # Создание датасета
    print(f"Loading evaluation dataset from {eval_dataset_path}...")
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_examples = json.load(f)
    max_length = (
        cfg.model.model.training.max_length
        if "training" in cfg.model.model
        else cfg.evaluation.max_length
    )
    eval_dataset = SecurityDataset(eval_examples, tokenizer, max_length)
    
    # Инициализация оценщика
    evaluator = SecurityModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        output_dir=cfg.paths.evaluation_dir
    )
    
    # Запуск оценки
    print("Starting evaluation...")
    metrics = evaluator.evaluate(eval_dataset)
    
    # Вывод результатов
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Генерация ответов на примерах
    print("\nGenerating responses for examples...")
    responses = evaluator.generate_responses(eval_examples)

    # Расчет и логирование специализированных метрик
    security_metrics = evaluator.evaluate_security_metrics(eval_examples, responses)
    metrics.update(security_metrics)
    evaluator.save_metrics(metrics)
    
    print(f"\nResults saved to {cfg.paths.evaluation_dir}")
    evaluator.close()
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 