# Pre-train / SFT

Pre-train в этом репозитории означает supervised instruction fine-tuning на подготовленном security dataset.

Поддерживаемые методы:

- `lora` — стандартный PEFT;
- `qlora` — 4-bit base model + LoRA;
- `prompt_tuning` — virtual prompt tokens;
- `p_tuning` — prompt encoder через PEFT `PromptEncoderConfig`;
- `full` — full fine-tuning через `peft=base`.

Любой метод выбирается независимо от модели:

```bash
python src/train.py model=gpt2 peft=lora
python src/train.py model=qwen2_5 peft=qlora
python src/train.py model=phi3 peft=prompt_tuning
python src/train.py model=mistral peft=p_tuning
python src/train.py model=gpt2 peft=base
```

Основной Trainer: `src/training/trainer.py`. Он создаёт tokenizer/model, PEFT adapter, `TrainingArguments`, TensorBoard и MLflow callbacks. Результат сохраняется в `models/<model>/checkpoints/<run_id>`.

Для smoke-прогона используйте `training.training.max_steps=2`, маленький dataset и отключённый MLflow только если он не установлен.

Полная матрица и pilot limits задаются в `configs/runtime.yaml`. Launcher явно
передает `max_steps`, `train_limit`, `eval_limit`, batch size и gradient
accumulation в Hydra; это не зависит от shell environment. После каждого
успешного pre-train автоматически запускается тот же stage benchmark.

Для full fine-tuning pilot-профиль использует `paged_adamw_8bit`: это уменьшает
память optimizer state и позволяет запускать Phi-3 на одной A100 40 GiB. Для
многокарточного production-запуска optimizer следует выбирать вместе с
настройкой FSDP.
