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
try:
    import wandb  # noqa: WPS433
except ImportError:
    wandb = None
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from .dataset import SecurityDataset
from .callbacks import PerplexityCallback
from .utils import setup_model_for_training, setup_tokenizer, ensure_output_dir
from .policy import PolicyOptimTrainer, create_reference_model

class SecurityModelTrainer:
    """Тренер для обучения модели безопасности с поддержкой различных методов fine-tuning."""
    
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
        if self.cfg.peft.peft.enabled:
            self.model = self._setup_peft()
        
        # Инициализация TensorBoard
        self.writer = SummaryWriter(log_dir=self.cfg.paths.tensorboard_dir)
    
    def _setup_peft(self) -> Union[PeftModel, AutoModelForCausalLM]:
        """Настройка PEFT адаптера в зависимости от выбранного метода."""
        method = self.cfg.peft.peft.method.lower()
        
        if method == "lora":
            config = LoraConfig(
                r=self.cfg.peft.peft.lora.r,
                lora_alpha=self.cfg.peft.peft.lora.alpha,
                target_modules=list(self.cfg.peft.peft.lora.target_modules),
                lora_dropout=self.cfg.peft.peft.lora.lora_dropout,
                bias=self.cfg.peft.peft.lora.bias,
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=self.cfg.peft.peft.lora.modules_to_save,
                inference_mode=False
            )
        
        elif method == "qlora":
            # Настройка квантизации для QLoRA
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            # Применяем квантизацию к модели
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
            config = LoraConfig(
                r=self.cfg.peft.peft.lora.r,
                lora_alpha=self.cfg.peft.peft.lora.alpha,
                target_modules=list(self.cfg.peft.peft.lora.target_modules),
                lora_dropout=self.cfg.peft.peft.lora.lora_dropout,
                bias=self.cfg.peft.peft.lora.bias,
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=self.cfg.peft.peft.lora.modules_to_save,
                inference_mode=False
            )
        
        elif method == "prefix_tuning":
            config = PrefixTuningConfig(
                num_virtual_tokens=self.cfg.peft.peft.prefix_tuning.num_virtual_tokens,
                encoder_hidden_size=self.cfg.peft.peft.prefix_tuning.encoder_hidden_size,
                prefix_projection=self.cfg.peft.peft.prefix_tuning.prefix_projection,
                task_type=TaskType.CAUSAL_LM
            )
        
        elif method == "prompt_tuning":
            config = PromptTuningConfig(
                num_virtual_tokens=self.cfg.peft.peft.prompt_tuning.num_virtual_tokens,
                prompt_tuning_init=self.cfg.peft.peft.prompt_tuning.prompt_tuning_init,
                token_dim=self.cfg.peft.peft.prompt_tuning.token_dim,
                prompt_tuning_init_text=self.cfg.peft.peft.prompt_tuning.prompt_tuning_init_text,
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
        report_to = []
        if self.cfg.logging.tensorboard.enabled:
            report_to.append("tensorboard")
        if self.cfg.logging.wandb.enabled and wandb is not None:
            report_to.append("wandb")
        elif self.cfg.logging.wandb.enabled:
            print("logging.wandb.enabled=true but wandb is not installed; skipping Weights & Biases reporting.")

        fsdp = None
        fsdp_config = None
        fsdp_cfg = self.cfg.training.training.get("fsdp", {})
        if fsdp_cfg.get("enabled", False):
            strategy = fsdp_cfg.get("sharding_strategy", "full_shard")
            fsdp = f"{strategy} auto_wrap" if fsdp_cfg.get("auto_wrap", True) else strategy
            fsdp_config = {
                "backward_prefetch": fsdp_cfg.get("backward_prefetch", "backward_pre"),
                "state_dict_type": fsdp_cfg.get("state_dict_type", "full_state_dict"),
                "use_orig_params": fsdp_cfg.get("use_orig_params", True),
                "sync_module_states": fsdp_cfg.get("sync_module_states", True),
                "cpu_offload": fsdp_cfg.get("cpu_offload", False),
                "min_num_params": fsdp_cfg.get("min_num_params", 0)
            }
            layer_cls = fsdp_cfg.get("transformer_layer_cls", [])
            if layer_cls:
                fsdp_config["transformer_layer_cls_to_wrap"] = list(layer_cls)
            mixed_precision = fsdp_cfg.get("mixed_precision")
            if mixed_precision in {"fp16", "bf16"}:
                fsdp_config["mixed_precision"] = mixed_precision

        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.cfg.training.training.num_train_epochs,
            per_device_train_batch_size=self.cfg.model.model.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.model.model.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.cfg.model.model.training.gradient_accumulation_steps,
            learning_rate=self.cfg.model.model.training.learning_rate,
            weight_decay=self.cfg.training.training.weight_decay,
            warmup_steps=self.cfg.training.training.warmup_steps,
            max_grad_norm=self.cfg.training.training.max_grad_norm,
            lr_scheduler_type=self.cfg.training.training.lr_scheduler_type,
            
            # Настройки логирования
            logging_dir=self.cfg.paths.training_logs_dir,
            logging_steps=self.cfg.logging.logging.logging_steps,
            logging_first_step=self.cfg.logging.logging.logging_first_step,
            report_to=report_to,
            
            # Настройки оценки
            eval_strategy=self.cfg.training.training.evaluation.eval_strategy,
            eval_steps=self.cfg.training.training.evaluation.eval_steps,
            metric_for_best_model=self.cfg.training.training.evaluation.metric_for_best_model,
            greater_is_better=self.cfg.training.training.evaluation.greater_is_better,
            load_best_model_at_end=self.cfg.training.training.evaluation.load_best_model_at_end,
            
            # Настройки сохранения
            save_strategy=self.cfg.training.training.save.save_strategy,
            save_steps=self.cfg.training.training.save.save_steps,
            save_total_limit=self.cfg.training.training.save.save_total_limit,
            
            # Оптимизация
            fp16=self.cfg.training.training.optimization.fp16 and torch.cuda.is_available(),
            gradient_checkpointing=self.cfg.training.training.optimization.gradient_checkpointing,
            dataloader_num_workers=self.cfg.training.training.optimization.dataloader_num_workers,
            dataloader_pin_memory=self.cfg.training.training.optimization.dataloader_pin_memory,
            
            # Дополнительные настройки
            seed=self.cfg.training.training.seed,
            remove_unused_columns=False,  # Важно для работы с нашим датасетом
            fsdp=fsdp,
            fsdp_config=fsdp_config
        )
    
    def train(
        self,
        train_dataset: DatasetDict,
        eval_dataset: Optional[DatasetDict] = None
    ):
        """Обучение модели."""
        training_method = self.cfg.training.training.get("method", "sft").lower()
        fsdp_enabled = self.cfg.training.training.get("fsdp", {}).get("enabled", False)
        if fsdp_enabled and self.cfg.peft.peft.enabled and self.cfg.peft.peft.method == "qlora":
            print("Warning: FSDP with QLoRA may be unstable. Consider disabling FSDP or QLoRA.")
        if self.cfg.logging.wandb.enabled and wandb is not None:
            wandb.init(
                project=self.cfg.logging.wandb.project,
                name=self.cfg.logging.wandb.name,
                config={
                    "model_name": self.model_name,
                    "peft_method": self.cfg.peft.peft.method if self.cfg.peft.peft.enabled else "full",
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
        
        trainer_cls = Trainer
        ref_model = None
        if training_method in {"grpo", "gspo"}:
            trainer_cls = PolicyOptimTrainer
            ref_model = create_reference_model(self.model_name, self.cfg)

        trainer = trainer_cls(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset["train"],
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
            callbacks=callbacks,
            ref_model=ref_model,
            policy_method=training_method,
            policy_config=self.cfg.training.training.get("policy_optimization", {})
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
        if self.cfg.peft.peft.enabled:
            trainer.model.save_pretrained(os.path.join(self.output_dir, f"{self.cfg.peft.peft.method}_adapter"))
        else:
            self.model.save_pretrained(os.path.join(self.output_dir, "full_model"))
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Закрываем TensorBoard writer
        self.writer.close()
        
        if self.cfg.logging.wandb.enabled and wandb is not None:
            wandb.finish()
    
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