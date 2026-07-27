# RL alignment

После pre-train можно продолжить обучение тем же base model/checkpoint одним из алгоритмов:

- `grpo`;
- `gspo`.

Текущая реализация находится в `src/training/policy.py`. Она использует reference model и KL-регуляризацию; `gspo` применяет симметричный KL, `grpo` — направленный KL. Это исследовательская policy-optimization реализация, а не готовая production RLHF-система с отдельной reward model.

RL получает pre-train artifact через `peft.peft.init_adapter_path` для PEFT или `model.model.pretrained_path` для full checkpoint:

```bash
python src/train.py \
  model=gpt2 peft=lora \
  training.training.method=gspo \
  peft.peft.init_adapter_path=models/gpt2/checkpoints/<pretrain>/lora_adapter
```

Для улучшения качества reward следует добавить отдельную reward model или явный `compute_reward` и зарегистрировать её в pipeline. Текущий код фиксирует прозрачный baseline KL-alignment и логирует его loss в TensorBoard/MLflow.
