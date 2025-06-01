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
    device_map: str = "auto"
) -> AutoModelForCausalLM:
    """Подготовка модели для обучения с учетом квантизации."""
    # Настройка квантизации если требуется
    quantization_config = None
    if cfg.model.model.load_in_4bit:
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