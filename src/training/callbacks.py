"""
Callbacks for model training.
"""

import math
import time
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

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

class TensorBoardCallback(TrainerCallback):
    """Callback для логирования метрик в TensorBoard."""
    
    def __init__(self, writer: SummaryWriter):
        self.writer = writer
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, state.global_step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"eval/{key}", value, state.global_step)

__all__ = ['PerplexityCallback', 'DetailedLoggingCallback', 'TensorBoardCallback'] 