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
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time  # Добавляем импорт в начало файла

class PerplexityCallback(TrainerCallback):
    """Callback для вычисления перплексии."""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            perplexity = math.exp(metrics["eval_loss"])
            metrics["eval_perplexity"] = perplexity

class DetailedLoggingCallback(TrainerCallback):
    """Callback для подробного логирования процесса обучения."""
    
    def __init__(self):
        self.start_time = None
        self.step_start_time = None
        self.total_steps = None
        self.current_epoch = 0
        self.steps_per_epoch = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n=== Начало обучения ===")
        print(f"Всего шагов: {state.max_steps}")
        print(f"Размер батча: {args.per_device_train_batch_size}")
        print(f"Накопление градиентов: {args.gradient_accumulation_steps}")
        print(f"Всего батчей на эпоху: {len(kwargs['train_dataloader'])}")
        print("=====================\n")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch += 1
        self.step_start_time = time.time()
        print(f"\n=== Эпоха {self.current_epoch} ===")
    
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            elapsed = time.time() - self.step_start_time
            print(f"\nШаг {state.global_step}/{state.max_steps} (эпоха {self.current_epoch})")
            print(f"Время с начала шага: {elapsed:.2f}с")
            print(f"Время с начала обучения: {(time.time() - self.start_time)/60:.2f}мин")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            metrics = kwargs.get("metrics", {})
            if metrics:
                print(f"Loss: {metrics.get('loss', 'N/A'):.4f}")
                if "learning_rate" in metrics:
                    print(f"Learning rate: {metrics['learning_rate']:.2e}")
            print(f"Скорость: {args.per_device_train_batch_size / (time.time() - self.step_start_time):.2f} батчей/с")
            self.step_start_time = time.time()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.step_start_time
        print(f"\n=== Завершение эпохи {self.current_epoch} ===")
        print(f"Время эпохи: {epoch_time/60:.2f}мин")
        print(f"Общее время: {(time.time() - self.start_time)/60:.2f}мин")
        print("=====================\n")

class SecurityDataset(Dataset):
    """Датасет для обучения модели на примерах безопасности."""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Формируем текст для обучения
        text = f"""### Инструкция: {example['instruction']}

### Контекст: {example.get('context', '')}

### Вход: {example['input']}

### Ответ: {example['output']}"""
        
        # Токенизируем текст
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }

class SecurityModelTrainer:
    """Тренер для обучения модели безопасности с поддержкой различных методов PEFT."""
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        cfg: DictConfig
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.cfg = cfg
        
        # Инициализация токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Инициализация модели
        self.model = self._setup_model()
        
        # Инициализация TensorBoard
        self.writer = SummaryWriter(log_dir=self.cfg.paths.tensorboard_dir)
    
    def _setup_model(self) -> PreTrainedModel:
        """Initialize the model with proper training setup."""
        print(f"Loading model from {self.cfg.model.model.name}...")
        
        # Определяем устройство для модели
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load model with proper configuration
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.model.name,
            torch_dtype=getattr(torch, self.cfg.model.model.torch_dtype),
            device_map=None,  # Changed from "auto" to None
            trust_remote_code=self.cfg.model.model.trust_remote_code,
            use_cache=False  # Required for gradient checkpointing
        )
        
        # Move model to device
        model = model.to(device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.cfg.training.training.optimization.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Set model to training mode
        model.train()
        
        # Configure PEFT if enabled
        if self.cfg.peft.peft.enabled:
            print(f"Setting up {self.cfg.peft.peft.method}...")
            model = self._setup_peft(model)
        
        return model
    
    def _setup_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """Configure PEFT for the model."""
        if self.cfg.peft.peft.method == "lora":
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=self.cfg.peft.peft.common.task_type,
                inference_mode=False,
                r=self.cfg.peft.peft.lora.r,
                lora_alpha=self.cfg.peft.peft.lora.alpha,
                lora_dropout=self.cfg.peft.peft.lora.lora_dropout,
                target_modules=self.cfg.peft.peft.lora.target_modules,
                bias=self.cfg.peft.peft.lora.bias,
                modules_to_save=self.cfg.peft.peft.lora.modules_to_save
            )
            
            # Apply LoRA to model
            model = get_peft_model(model, peft_config)
            
            # Ensure LoRA parameters are trainable
            for name, param in model.named_parameters():
                if "lora" in name:
                    param.requires_grad_(True)
            
            # Print parameter statistics
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.2f}%")
            
            return model
        else:
            raise ValueError(f"Unsupported PEFT method: {self.cfg.peft.peft.method}")
    
    def _setup_training_args(self) -> TrainingArguments:
        """Configure training arguments."""
        # Определяем устройство для обучения
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return TrainingArguments(
            output_dir=self.cfg.paths.checkpoints_dir,
            num_train_epochs=self.cfg.training.training.num_train_epochs,
            per_device_train_batch_size=self.cfg.training.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.training.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.cfg.training.training.gradient_accumulation_steps,
            learning_rate=self.cfg.training.training.learning_rate,
            weight_decay=self.cfg.training.training.weight_decay,
            warmup_steps=self.cfg.training.training.warmup_steps,
            max_grad_norm=self.cfg.training.training.max_grad_norm,
            lr_scheduler_type=self.cfg.training.training.lr_scheduler_type,
            
            # Device settings
            no_cuda=not torch.cuda.is_available(),
            
            # Logging settings
            logging_dir=self.cfg.paths.training_logs_dir,
            logging_steps=self.cfg.training.training.logging.logging_steps,
            logging_first_step=self.cfg.training.training.logging.logging_first_step,
            report_to=self.cfg.training.training.logging.report_to,
            
            # Evaluation settings
            eval_strategy=self.cfg.training.training.evaluation.evaluation_strategy,
            eval_steps=self.cfg.training.training.evaluation.eval_steps,
            metric_for_best_model=self.cfg.training.training.evaluation.metric_for_best_model,
            greater_is_better=self.cfg.training.training.evaluation.greater_is_better,
            load_best_model_at_end=self.cfg.training.training.evaluation.load_best_model_at_end,
            
            # Save settings
            save_strategy=self.cfg.training.training.save.save_strategy,
            save_steps=self.cfg.training.training.save.save_steps,
            save_total_limit=self.cfg.training.training.save.save_total_limit,
            
            # Optimization settings
            fp16=self.cfg.training.training.optimization.fp16 and torch.cuda.is_available(),
            gradient_checkpointing=self.cfg.training.training.optimization.gradient_checkpointing,
            dataloader_num_workers=self.cfg.training.training.optimization.dataloader_num_workers,
            dataloader_pin_memory=self.cfg.training.training.optimization.dataloader_pin_memory,
            
            # Additional settings
            seed=self.cfg.training.training.seed,
            remove_unused_columns=False
        )
    
    def train(
        self,
        train_dataset: DatasetDict,
        eval_dataset: Optional[DatasetDict] = None
    ):
        """Обучение модели."""
        if self.cfg.training.training.logging.wandb.enabled:
            wandb.init(
                project=self.cfg.training.training.logging.wandb.project,
                name=self.cfg.training.training.logging.wandb.name,
                config={
                    "model_name": self.model_name,
                    "peft_method": self.cfg.peft.peft.method if self.cfg.peft.peft.enabled else "full",
                    **OmegaConf.to_container(self.cfg.model, resolve=True),
                    **OmegaConf.to_container(self.cfg.training.training, resolve=True),
                    **OmegaConf.to_container(self.cfg.peft.peft, resolve=True)
                }
            )
        
        training_args = self._setup_training_args()
        
        # Добавляем callbacks для логирования
        callbacks = [
            PerplexityCallback(),
            DetailedLoggingCallback()
        ]
        
        # Если eval_dataset не передан, отключаем оценку
        if eval_dataset is None:
            training_args.eval_strategy = "no"
            eval_dataset = None
        else:
            eval_dataset = eval_dataset["validation"]
        
        # Устанавливаем более частые логи
        training_args.logging_steps = 1  # Логируем каждый шаг
        training_args.logging_first_step = True
        training_args.output_dir = self.cfg.paths.checkpoints_dir  # Обновляем путь для сохранения чекпоинтов
        training_args.logging_dir = self.cfg.paths.training_logs_dir  # Обновляем путь для логов
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset["train"],
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
            callbacks=callbacks
        )
        
        # Логируем информацию о модели
        print("\n=== Информация о модели ===")
        print(f"Размер модели: {sum(p.numel() for p in self.model.parameters()):,} параметров")
        print(f"Обучаемые параметры: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Размер батча: {training_args.per_device_train_batch_size}")
        print(f"Накопление градиентов: {training_args.gradient_accumulation_steps}")
        print(f"Эффективный размер батча: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"Путь для чекпоинтов: {training_args.output_dir}")
        print(f"Путь для логов: {training_args.logging_dir}")
        print("========================\n")
        
        self.writer.add_text("model_config", str(self.model.config))
        self.writer.add_text("training_config", str(training_args))
        
        # Запускаем обучение
        print("Starting training...")
        trainer.train()
        
        # Сохраняем финальные метрики только если был eval_dataset
        if eval_dataset is not None:
            final_metrics = trainer.evaluate()
            print("\n=== Финальные метрики ===")
            for metric_name, value in final_metrics.items():
                print(f"{metric_name}: {value:.4f}")
                self.writer.add_scalar(f"final/{metric_name}", value)
            print("======================\n")
        
        # Сохранение модели
        print("\n=== Сохранение модели ===")
        if self.cfg.peft.peft.enabled:
            save_path = os.path.join(self.cfg.paths.adapters_dir, f"{self.cfg.peft.peft.method}_adapter")
            print(f"Сохранение адаптера в {save_path}")
            trainer.model.save_pretrained(save_path)
        else:
            save_path = os.path.join(self.cfg.paths.checkpoints_dir, "full_model")
            print(f"Сохранение полной модели в {save_path}")
            trainer.model.save_pretrained(save_path)
        
        print(f"Сохранение токенизатора в {self.cfg.paths.tokenizer_dir}")
        self.tokenizer.save_pretrained(self.cfg.paths.tokenizer_dir)
        print("========================\n")
        
        # Закрываем TensorBoard writer
        self.writer.close()
        
        if self.cfg.training.training.logging.wandb.enabled:
            wandb.finish()
        
        print("\n=== Обучение завершено ===")
        print(f"Общее время обучения: {(time.time() - self.start_time)/60:.2f}мин")
        print("========================\n")
    
    def save_model(self, path: str):
        """Сохранение модели."""
        if self.cfg.peft.peft.enabled:
            self.model.save_pretrained(os.path.join(path, f"{self.cfg.peft.peft.method}_adapter"))
        else:
            self.model.save_pretrained(os.path.join(path, "full_model"))
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def load_model(
        cls,
        path: str,
        cfg: DictConfig
    ) -> "SecurityModelTrainer":
        """Загрузка сохраненной модели."""
        trainer = cls(cfg.model.model.name, path, cfg)
        
        if cfg.peft.peft.enabled:
            trainer.model = PeftModel.from_pretrained(
                trainer.model,
                os.path.join(path, f"{cfg.peft.peft.method}_adapter")
            )
        else:
            trainer.model = AutoModelForCausalLM.from_pretrained(
                os.path.join(path, "full_model"),
                torch_dtype=getattr(torch, cfg.model.model.torch_dtype),
                device_map=cfg.model.model.device_map
            )
        
        trainer.tokenizer = AutoTokenizer.from_pretrained(path)
        return trainer

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Загружаем конфигурацию
    print("Loading configuration...")
    print(OmegaConf.to_yaml(cfg))
    
    # Создаем необходимые директории
    for dir_path in [
        cfg.paths.checkpoints_dir,
        cfg.paths.adapters_dir,
        cfg.paths.tokenizer_dir,
        cfg.paths.tensorboard_dir,
        cfg.paths.training_logs_dir
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Загружаем датасет
    print("Loading dataset...")
    dataset_path = Path(cfg.dataset.output_dir)
    
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
    
    train_dataset = SecurityDataset(train_examples, tokenizer, cfg.training.training.max_length)
    eval_dataset = SecurityDataset(eval_examples, tokenizer, cfg.training.training.max_length)
    
    # Создаем DatasetDict
    datasets = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset
    })
    
    # Инициализируем тренер
    trainer = SecurityModelTrainer(
        model_name=cfg.model.model.name,
        output_dir=cfg.paths.checkpoints_dir,
        cfg=cfg
    )
    
    # Запускаем обучение
    print("Starting training...")
    trainer.train(datasets)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 