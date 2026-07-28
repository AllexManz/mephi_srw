# RL alignment

После pre-train можно продолжить обучение тем же base model/checkpoint одним из алгоритмов:

- `grpo`;
- `gspo`.

Текущая реализация находится в `src/training/policy.py`. Она использует
reference model и KL-регуляризацию; `gspo` применяет симметричный KL, `grpo` —
направленный KL. Это внутренние policy-optimization baselines, а не канонические
rollout-group GRPO/GSPO и не production RLHF-система с reward model. Отчеты и
MLflow tags не должны интерпретировать их как полноценное RLHF.

RL получает pre-train artifact через `peft.peft.init_adapter_path` для PEFT или `model.model.pretrained_path` для full checkpoint:

```bash
python src/train.py \
  model=gpt2 peft=lora \
  training.training.method=gspo \
  peft.peft.init_adapter_path=models/gpt2/checkpoints/<pretrain>/lora_adapter
```

Для улучшения качества reward следует добавить отдельную reward model или явный `compute_reward` и зарегистрировать её в pipeline. Текущий код фиксирует прозрачный baseline KL-alignment и логирует его loss в TensorBoard/MLflow.

Campaign runner создает независимую ветку для каждой пары
`pretrain_method/rl_method`, всегда начиная от соответствующего pre-train
artifact, и запускает benchmark после каждой RL-стадии. Неуспешная ветка не
повреждает checkpoint другой пары.

PEFT сохраняет prompt/prefix adapters как итоговые virtual-token embeddings и
не разрешает загружать их с `is_trainable=True`. Framework загружает такой
adapter штатно, замораживает base model и явно продолжает оптимизацию только
параметров `prompt_encoder`; LoRA/QLoRA загружаются обычным trainable-режимом.
Перед KL policy logits для virtual-token методов выравниваются справа с
reference logits, поэтому prompt tokens не сравниваются с реальными токенами.

Для full fine-tuning на одной A100 40 GiB reference model загружается в NF4
согласно `training.policy_optimization.full_reference_load_in_4bit`; policy
checkpoint остаётся BF16. Это осознанный memory/точность компромисс pilot-
профиля. Для публикационного сравнения на нескольких GPU reference следует
оставлять в BF16 и использовать FSDP.
