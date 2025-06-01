from dataclasses import dataclass
from typing import Optional, List
import os
from pathlib import Path

@dataclass
class ModelConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"  # Базовая модель
    max_length: int = 2048
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    output_dir: str = "models/checkpoints"
    
    # LoRA параметры
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = ["q_proj", "v_proj"]

@dataclass
class DataConfig:
    train_data_path: str = "data/processed/train.json"
    eval_data_path: str = "data/processed/eval.json"
    test_data_path: str = "data/processed/test.json"
    max_samples: Optional[int] = None

@dataclass
class TrainingConfig:
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    use_wandb: bool = True
    wandb_project: str = "security-llm"
    wandb_run_name: Optional[str] = None

@dataclass
class SIEMConfig:
    # Elasticsearch конфигурация (пример для ELK Stack)
    es_host: str = "localhost"
    es_port: int = 9200
    es_username: Optional[str] = None
    es_password: Optional[str] = None
    es_index_pattern: str = "security-*"
    
    # Параметры для анализа событий
    max_events_per_query: int = 1000
    time_window_minutes: int = 60
    alert_threshold: float = 0.8

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    siem: SIEMConfig = SIEMConfig()
    
    def __post_init__(self):
        # Создаем необходимые директории
        os.makedirs(self.model.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.data.train_data_path), exist_ok=True)
        
        # Устанавливаем wandb run name если не задан
        if self.training.wandb_run_name is None:
            self.training.wandb_run_name = f"{Path(self.model.model_name).name}-{self.model.lora_r}"

# Создаем глобальный конфиг
config = Config() 