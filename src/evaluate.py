"""
Script for running model evaluation using the existing evaluator module.
"""

import os
import hydra
from pathlib import Path
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from evaluation.evaluator import SecurityModelEvaluator
from training.dataset import SecurityDataset

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    # Определение путей
    if cfg.peft.peft.enabled:
        model_path = os.path.join(cfg.paths.adapters_dir, f"{cfg.peft.peft.method}_adapter")
    else:
        model_path = os.path.join(cfg.paths.checkpoints_dir, "full_model")
    
    tokenizer_path = cfg.paths.tokenizer_dir
    eval_dataset_path = os.path.join(cfg.paths.dataset_dir, "eval.json")
    
    # Проверка существования файлов
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    if not os.path.exists(eval_dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_dataset_path}")
    
    # Загрузка токенизатора
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Загрузка модели
    print(f"Loading model from {model_path}...")
    if cfg.peft.peft.enabled:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model.name,
            torch_dtype=getattr(torch, cfg.model.model.torch_dtype),
            device_map=None,
            trust_remote_code=cfg.model.model.trust_remote_code
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, cfg.model.model.torch_dtype),
            device_map=None,
            trust_remote_code=cfg.model.model.trust_remote_code
        )
    
    # Создание датасета
    print(f"Loading evaluation dataset from {eval_dataset_path}...")
    eval_dataset = SecurityDataset.from_json(eval_dataset_path, tokenizer)
    
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
    examples = eval_dataset.get_examples()
    responses = evaluator.generate_responses(examples)
    
    print(f"\nResults saved to {cfg.paths.evaluation_dir}")
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 