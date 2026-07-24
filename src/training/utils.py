"""
Utility functions for model training.
"""

import os
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

def setup_model_for_training(
    model_name: str,
    cfg: dict,
    device_map: Optional[str] = None
) -> AutoModelForCausalLM:
    """Подготовка модели для обучения с учетом квантизации."""
    # Не подменяем настройку модели значением ``auto``: для CPU и FSDP
    # transformers должен получить ``None``, иначе модель может быть
    # принудительно разложена через accelerate до старта Trainer.
    if device_map is None:
        device_map = cfg.model.model.get("device_map")
    fsdp_enabled = cfg.training.training.get("fsdp", {}).get("enabled", False)
    if fsdp_enabled:
        device_map = None
    # Настройка квантизации если требуется
    quantization_config = None
    if (
        cfg.peft.peft.method == "qlora"
        and "quantization" in cfg.peft.peft
        and cfg.peft.peft.quantization.load_in_4bit
    ):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(
                torch,
                cfg.peft.peft.quantization.bnb_4bit_compute_dtype
            ),
            bnb_4bit_quant_type=cfg.peft.peft.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.peft.peft.quantization.bnb_4bit_use_double_quant
        )
    elif cfg.model.model.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif cfg.model.model.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    # Загрузка модели
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, cfg.model.model.torch_dtype),
        device_map=device_map,
        trust_remote_code=cfg.model.model.trust_remote_code,
        quantization_config=quantization_config
    )

    # Убедимся, что модель в режиме обучения
    model.train()

    # Подготовка модели для квантизации если требуется
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
        # Явно включаем градиенты для параметров
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad_(True)

    return model

def setup_tokenizer(model_name: str) -> AutoTokenizer:
    """Подготовка токенизатора."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def ensure_output_dir(output_dir: str) -> Path:
    """Создание директории для сохранения модели если она не существует."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def update_paths_with_run_id(cfg) -> str:
    """
    Генерирует уникальный run_id формата: (бейзлайн модель)_(метод обучения)_(дата и время начала)
    и динамически обновляет все пути в конфигурации для использования этого run_id.
    """
    import datetime

    # Получаем ID базовой модели
    model_id = cfg.model.model.id

    # Получаем метод обучения
    train_method = cfg.training.training.method

    # Получаем суффикс PEFT
    peft_suffix = f"_{cfg.peft.peft.method}" if cfg.peft.peft.enabled else "_full"

    # Форматируем временную метку
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Генерируем Run ID
    run_id = f"{model_id}_{train_method}{peft_suffix}_{timestamp}"

    # Базовая директория
    base_dir = cfg.paths.base_dir

    # Обновляем пути в конфигурации
    cfg.paths.checkpoints_dir = os.path.join(base_dir, "models", model_id, "checkpoints", run_id)
    cfg.paths.adapters_dir = os.path.join(base_dir, "models", model_id, "adapters", run_id)
    cfg.paths.tokenizer_dir = os.path.join(base_dir, "models", model_id, "tokenizer", run_id)

    cfg.paths.logs_dir = os.path.join(base_dir, "logs", model_id, run_id)
    cfg.paths.tensorboard_dir = os.path.join(cfg.paths.logs_dir, "tensorboard")
    cfg.paths.training_logs_dir = os.path.join(cfg.paths.logs_dir, "training")
    cfg.paths.evaluation_dir = os.path.join(cfg.paths.logs_dir, "evaluation")

    return run_id
