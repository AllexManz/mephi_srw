"""
Main trainer module for security language model.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)
from datasets import DatasetDict
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from .dataset import SecurityDataset
from .callbacks import PerplexityCallback
from .utils import setup_model_for_training, setup_tokenizer, ensure_output_dir

class SecurityModelTrainer:
    """Тренер для обучения модели безопасности с поддержкой различных методов PEFT."""
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        cfg: DictConfig
    ):
        self.model_name = model_name
        self.output_dir = ensure_output_dir(output_dir)
        self.cfg = cfg
        
        # Инициализация токенизатора
        self.tokenizer = setup_tokenizer(model_name)
        
        # Инициализация модели
        self.model = setup_model_for_training(model_name, cfg)
        
        # Настройка PEFT если требуется
        if self.cfg.peft.enabled:
            self.model = self._setup_peft()
        
        # Инициализация TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    
    def _setup_peft(self) -> Union[PeftModel, AutoModelForCausalLM]:
        """Настройка PEFT адаптера в зависимости от выбранного метода."""
        method = self.cfg.peft.method.lower()
        
        if method == "lora":
            config = LoraConfig(
                r=self.cfg.peft.lora.r,
                lora_alpha=self.cfg.peft.lora.alpha,
                target_modules=self.cfg.peft.lora.target_modules,
                lora_dropout=self.cfg.peft.lora.lora_dropout,
                bias=self.cfg.peft.lora.bias,
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=self.cfg.peft.lora.modules_to_save,
                inference_mode=False
            )
        
        elif method == "qlora":
            config = LoraConfig(
                r=self.cfg.peft.qlora.r,
                lora_alpha=self.cfg.peft.qlora.alpha,
                target_modules=self.cfg.peft.qlora.target_modules,
                lora_dropout=self.cfg.peft.qlora.lora_dropout,
                bias=self.cfg.peft.qlora.bias,
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=self.cfg.peft.qlora.modules_to_save,
                inference_mode=False
            )
        
        elif method == "prefix_tuning":
            config = PrefixTuningConfig(
                num_virtual_tokens=self.cfg.peft.prefix_tuning.num_virtual_tokens,
                encoder_hidden_size=self.cfg.peft.prefix_tuning.encoder_hidden_size,
                prefix_projection=self.cfg.peft.prefix_tuning.prefix_projection,
                task_type=TaskType.CAUSAL_LM
            )
        
        elif method == "prompt_tuning":
            config = PromptTuningConfig(
                num_virtual_tokens=self.cfg.peft.prompt_tuning.num_virtual_tokens,
                prompt_tuning_init=self.cfg.peft.prompt_tuning.prompt_tuning_init,
                token_dim=self.cfg.peft.prompt_tuning.token_dim,
                prompt_tuning_init_text=self.cfg.peft.prompt_tuning.prompt_tuning_init_text,
                task_type=TaskType.CAUSAL_LM
            )
        
        else:
            raise ValueError(
                f"Unknown PEFT method: {method}. "
                "Supported methods are: lora, qlora, prefix_tuning, prompt_tuning"
            )
        
        model = get_peft_model(self.model, config)
        
        # Проверяем, что параметры действительно обучаемые
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params:.2f}%")
        
        # Убедимся, что модель в режиме обучения
        model.train()
        
        return model
    
    def _setup_training_args(self) -> TrainingArguments:
        """Настройка параметров обучения."""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.cfg.training.num_train_epochs,
            per_device_train_batch_size=self.cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.cfg.training.gradient_accumulation_steps,
            learning_rate=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
            warmup_steps=self.cfg.training.warmup_steps,
            max_grad_norm=self.cfg.training.max_grad_norm,
            lr_scheduler_type=self.cfg.training.lr_scheduler_type,
            
            # Настройки логирования
            logging_dir=self.cfg.training.logging.logging_dir,
            logging_steps=self.cfg.training.logging.logging_steps,
            logging_first_step=self.cfg.training.logging.logging_first_step,
            report_to=self.cfg.training.logging.report_to,
            
            # Настройки оценки
            eval_strategy=self.cfg.training.evaluation.eval_strategy,
            eval_steps=self.cfg.training.evaluation.eval_steps,
            metric_for_best_model=self.cfg.training.evaluation.metric_for_best_model,
            greater_is_better=self.cfg.training.evaluation.greater_is_better,
            load_best_model_at_end=self.cfg.training.evaluation.load_best_model_at_end,
            
            # Настройки сохранения
            save_strategy=self.cfg.training.save.save_strategy,
            save_steps=self.cfg.training.save.save_steps,
            save_total_limit=self.cfg.training.save.save_total_limit,
            
            # Оптимизация
            fp16=self.cfg.training.optimization.fp16,
            gradient_checkpointing=self.cfg.training.optimization.gradient_checkpointing,
            dataloader_num_workers=self.cfg.training.optimization.dataloader_num_workers,
            dataloader_pin_memory=self.cfg.training.optimization.dataloader_pin_memory,
            
            # Дополнительные настройки
            seed=self.cfg.training.seed,
            remove_unused_columns=False,  # Важно для работы с нашим датасетом
        )
    
    def train(
        self,
        train_dataset: DatasetDict,
        eval_dataset: Optional[DatasetDict] = None
    ):
        """Обучение модели."""
        if self.cfg.logging.wandb.enabled:
            wandb.init(
                project=self.cfg.logging.wandb.project,
                name=self.cfg.logging.wandb.name,
                config={
                    "model_name": self.model_name,
                    "peft_method": self.cfg.peft.method if self.cfg.peft.enabled else "full",
                    **OmegaConf.to_container(self.cfg.model, resolve=True),
                    **OmegaConf.to_container(self.cfg.training, resolve=True),
                    **OmegaConf.to_container(self.cfg.peft, resolve=True)
                }
            )
        
        training_args = self._setup_training_args()
        
        # Добавляем callback для вычисления перплексии
        callbacks = [PerplexityCallback()]
        
        # Если eval_dataset не передан, отключаем оценку
        if eval_dataset is None:
            training_args.eval_strategy = "no"
            eval_dataset = None
        else:
            eval_dataset = eval_dataset["validation"]
        
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
        self.writer.add_text("model_config", str(self.model.config))
        self.writer.add_text("training_config", str(training_args))
        
        # Запускаем обучение
        trainer.train()
        
        # Сохраняем финальные метрики только если был eval_dataset
        if eval_dataset is not None:
            final_metrics = trainer.evaluate()
            for metric_name, value in final_metrics.items():
                self.writer.add_scalar(f"final/{metric_name}", value)
        
        # Сохранение модели
        if self.cfg.peft.enabled:
            trainer.model.save_pretrained(os.path.join(self.output_dir, f"{self.cfg.peft.method}_adapter"))
        else:
            trainer.model.save_pretrained(os.path.join(self.output_dir, "full_model"))
        
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Закрываем TensorBoard writer
        self.writer.close()
        
        if self.cfg.logging.wandb.enabled:
            wandb.finish()
    
    def save_model(self, path: str):
        """Сохранение модели."""
        if self.cfg.peft.enabled:
            self.model.save_pretrained(os.path.join(path, f"{self.cfg.peft.method}_adapter"))
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
        
        if cfg.peft.enabled:
            trainer.model = PeftModel.from_pretrained(
                trainer.model,
                os.path.join(path, f"{cfg.peft.method}_adapter")
            )
        else:
            trainer.model = AutoModelForCausalLM.from_pretrained(
                os.path.join(path, "full_model"),
                torch_dtype=getattr(torch, cfg.model.torch_dtype),
                device_map=cfg.model.device_map
            )
        
        trainer.tokenizer = AutoTokenizer.from_pretrained(path)
        return trainer 