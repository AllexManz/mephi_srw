import os
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
    PreTrainedModel
)
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
    PeftConfig
)
from datasets import DatasetDict
try:
    import wandb  # noqa: WPS433
except ImportError:  # optional dependency — training works with logging.wandb.enabled=false
    wandb = None
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time  # Добавляем импорт в начало файла

from training.dataset import SecurityDataset
from training.trainer import SecurityModelTrainer
from training.utils import update_paths_with_run_id
from training.mlflow_logging import start_run, end_run, log_artifacts

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    # Загружаем конфигурацию
    print("Loading configuration...")

    # Генерируем Run ID и динамически обновляем пути
    run_id = update_paths_with_run_id(cfg)
    print(f"Generated Run ID: {run_id}")
    print(OmegaConf.to_yaml(cfg))

    # Проверяем и создаем все необходимые директории
    required_dirs = [
        cfg.paths.models_dir,
        cfg.paths.checkpoints_dir,
        cfg.paths.adapters_dir,
        cfg.paths.tokenizer_dir,
        cfg.paths.logs_dir,
        cfg.paths.tensorboard_dir,
        cfg.paths.training_logs_dir,
        cfg.paths.evaluation_dir,
        cfg.paths.dataset_dir
    ]

    print("\nChecking and creating directories...")
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Directory exists: {path}")

    # Загружаем датасет
    print("\nLoading dataset...")
    dataset_path = Path(cfg.paths.dataset_dir)

    print(f"Looking for dataset files in: {dataset_path}")

    train_file = dataset_path / "train.json"
    eval_file = dataset_path / "eval.json"

    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset file not found at {train_file}")
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation dataset file not found at {eval_file}")

    with open(train_file, "r", encoding="utf-8") as f:
        train_examples = json.load(f)
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_examples = json.load(f)

    print(f"Loaded {len(train_examples)} training examples and {len(eval_examples)} evaluation examples")

    # Создаем датасеты
    print("Creating datasets...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model.name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = SecurityDataset(train_examples, tokenizer, cfg.model.model.training.max_length)
    eval_dataset = SecurityDataset(eval_examples, tokenizer, cfg.model.model.training.max_length)

    # Создаем DatasetDict
    datasets = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset
    })

    mlflow_run = start_run(
        cfg,
        run_id,
        train_examples=len(train_examples),
        eval_examples=len(eval_examples),
    )
    try:
        # Инициализируем тренер
        trainer = SecurityModelTrainer(
            model_name=cfg.model.model.name,
            output_dir=cfg.paths.checkpoints_dir,
            cfg=cfg
        )

        # Запускаем обучение
        print("Starting training...")
        trainer.train(datasets)
        log_artifacts(cfg.paths.checkpoints_dir, "model")
        print("Training completed!")
    except Exception:
        end_run(mlflow_run, status="FAILED")
        raise
    else:
        end_run(mlflow_run, status="FINISHED")

if __name__ == "__main__":
    main()
