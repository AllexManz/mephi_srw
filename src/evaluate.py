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
    checkpoints_base = Path(cfg.paths.base_dir) / "models" / cfg.model.model.id / "checkpoints"

    if cfg.peft.peft.enabled and cfg.peft.peft.adapter_path:
        model_path = cfg.peft.peft.adapter_path
        tokenizer_path = str(Path(model_path).parent)
    else:
        # Автоматическое определение последней запущенной сессии обучения
        if checkpoints_base.exists():
            run_dirs = [d for d in checkpoints_base.iterdir() if d.is_dir()]
            if run_dirs:
                latest_run = max(run_dirs, key=os.path.getmtime)
                run_dir = latest_run
                print(f"Auto-detected latest run directory: {run_dir}")

                if cfg.peft.peft.enabled:
                    model_path = os.path.join(run_dir, f"{cfg.peft.peft.method}_adapter")
                else:
                    model_path = os.path.join(run_dir, "full_model")
                tokenizer_path = str(run_dir)
            else:
                raise FileNotFoundError(f"No run directories found in {checkpoints_base}")
        else:
            raise FileNotFoundError(f"Checkpoints base directory {checkpoints_base} does not exist")

    # Обновляем директорию оценки, чтобы результаты сохранялись в папку конкретного запуска
    run_id = Path(tokenizer_path).name
    cfg.paths.evaluation_dir = os.path.join(cfg.paths.base_dir, "logs", cfg.model.model.id, run_id, "evaluation")
    print(f"Evaluation results will be saved to: {cfg.paths.evaluation_dir}")

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
    eval_cfg = cfg.evaluation.evaluation if "evaluation" in cfg.evaluation else cfg.evaluation
    gen_cfg = eval_cfg.generation if "generation" in eval_cfg else eval_cfg
    max_gen_examples = gen_cfg.get("max_examples")

    if max_gen_examples and len(eval_examples) > max_gen_examples:
        print(f"Limiting response generation to first {max_gen_examples} examples (out of {len(eval_examples)}) to save time.")
        gen_examples = eval_examples[:max_gen_examples]
    else:
        gen_examples = eval_examples

    responses = evaluator.generate_responses(gen_examples)

    # Расчет и логирование специализированных метрик
    security_metrics = evaluator.evaluate_security_metrics(gen_examples, responses)
    metrics.update(security_metrics)
    evaluator.save_metrics(metrics)

    print(f"\nResults saved to {cfg.paths.evaluation_dir}")
    evaluator.close()
    print("Evaluation completed!")

if __name__ == "__main__":
    main()