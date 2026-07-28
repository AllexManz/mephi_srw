# security-llm-pilot: отчет об экспериментальной кампании

## Аннотация

Отчет фиксирует воспроизводимую pilot-кампанию сравнения базовых security LLM, supervised fine-tuning и последующего policy alignment. Результаты автоматически собраны из неизменяемых JSON-артефактов; отсутствующие или упавшие прогоны не заменяются синтетическими значениями.

## Экспериментальная постановка

- Сгенерировано: `2026-07-28T00:26:13.291366+00:00`.
- Лучшая по baseline модель: `phi3`.
- Успешных стадий: `21`; упавших: `2`.
- Train steps: `10`; train/eval limits: `256`/`64`.
- Benchmark limit: `30`; max new tokens: `64`.
- Metric denominators: `{"accuracy": 5, "actionability": 20, "safety": 5, "word_overlap": 20}`.
- Environment: Python `3.10.12`, PyTorch `2.9.1`, Transformers `4.46.3`, PEFT `0.13.2`.
- Accelerator: `NVIDIA A100-PCIE-40GB (40 GiB)`; platform: `Linux-5.4.210-39.1.bert2-x86_64-with-glibc2.35`.
- Pre-train методы: `lora, qlora, prompt_tuning, p_tuning, full`.
- Alignment методы: `grpo, gspo`.

Главная сравнительная величина для автоматического выбора модели — среднее доступных quality-метрик accuracy, safety, word overlap и actionability. Latency в выбор не входит.

## Результаты базовых моделей

| Model | Status | Score | Accuracy | Safety | Overlap | Actionability | Latency, s |
|---|---|---|---|---|---|---|---|
| gpt2 | completed | 0.1160 | 0.4000 | 0.0000 | 0.0639 | 0.0000 | 0.5219 |
| mistral | completed | 0.2776 | 1.0000 | 0.0000 | 0.1104 | 0.0000 | 2.5066 |
| qwen2 | completed | 0.2432 | 0.8000 | 0.0000 | 0.1229 | 0.0500 | 1.5471 |
| qwen2_5 | completed | 0.3146 | 0.8000 | 0.2000 | 0.1085 | 0.1500 | 1.6347 |
| phi3 | completed | 0.3356 | 1.0000 | 0.2000 | 0.0922 | 0.0500 | 2.3384 |
| gemma2 | failed | — | — | — | — | — | — |
| llama3_2 | failed | — | — | — | — | — | — |
| deepseek_r1_1_5b | completed | 0.1862 | 0.4000 | 0.0000 | 0.0950 | 0.2500 | 1.4276 |

## Результаты обучения и alignment

| Variant | Stage | Status | Score | Δ baseline | Accuracy | Safety | Overlap | Actionability | Latency, s |
|---|---|---|---|---|---|---|---|---|---|
| full/grpo | rl | completed | 0.2928 | -0.0427 | 1.0000 | 0.0000 | 0.1212 | 0.0500 | 2.4102 |
| full/gspo | rl | completed | 0.2919 | -0.0437 | 1.0000 | 0.0000 | 0.1176 | 0.0500 | 2.3643 |
| full/pretrain | pretrain | completed | 0.2763 | -0.0592 | 1.0000 | 0.0000 | 0.1053 | 0.0000 | 2.3025 |
| lora/grpo | rl | completed | 0.3368 | 0.0012 | 1.0000 | 0.2000 | 0.0972 | 0.0500 | 2.6491 |
| lora/gspo | rl | completed | 0.3364 | 0.0009 | 1.0000 | 0.2000 | 0.0957 | 0.0500 | 2.7059 |
| lora/pretrain | pretrain | completed | 0.3360 | 0.0005 | 1.0000 | 0.2000 | 0.0941 | 0.0500 | 2.6929 |
| p_tuning/grpo | rl | completed | 0.3253 | -0.0103 | 0.8000 | 0.2000 | 0.1011 | 0.2000 | 2.3146 |
| p_tuning/gspo | rl | completed | 0.3757 | 0.0401 | 0.8000 | 0.4000 | 0.1028 | 0.2000 | 2.3060 |
| p_tuning/pretrain | pretrain | completed | 0.3088 | -0.0268 | 0.8000 | 0.2000 | 0.0852 | 0.1500 | 2.2996 |
| prompt_tuning/grpo | rl | completed | 0.2361 | -0.0994 | 0.8000 | 0.0000 | 0.0945 | 0.0500 | 2.3411 |
| prompt_tuning/gspo | rl | completed | 0.2477 | -0.0879 | 0.8000 | 0.0000 | 0.0908 | 0.1000 | 2.3389 |
| prompt_tuning/pretrain | pretrain | completed | 0.2352 | -0.1003 | 0.8000 | 0.0000 | 0.0909 | 0.0500 | 2.3424 |
| qlora/grpo | rl | completed | 0.3368 | 0.0012 | 1.0000 | 0.2000 | 0.0972 | 0.0500 | 2.6766 |
| qlora/gspo | rl | completed | 0.3385 | 0.0030 | 1.0000 | 0.2000 | 0.1040 | 0.0500 | 2.6754 |
| qlora/pretrain | pretrain | completed | 0.3366 | 0.0011 | 1.0000 | 0.2000 | 0.0966 | 0.0500 | 2.6997 |

## Основные наблюдения

- Победитель baseline: `phi3` со score `0.3356`.
- Лучший завершенный обученный вариант: `p_tuning/gspo` со score `0.3757`.
- Разности в таблице относятся к baseline выбранной модели; из-за pilot-размера их нельзя трактовать как статистически значимый эффект.

## Неуспешные стадии

| Scope | Variant | Reason | Diagnostic log |
|---|---|---|---|
| baseline | gemma2 | huggingface_hub.errors.GatedRepoError: 403 Client Error. (Request ID: Root=1-6a67e9a5-4180a0de0d3fc0bd4353c153;aa01dbc5-758f-47ce-8de3-bbaa573ba58d) | outputs/campaigns/security-llm-pilot/logs/baseline__gemma2.log |
| baseline | llama3_2 | huggingface_hub.errors.GatedRepoError: 403 Client Error. (Request ID: Root=1-6a67e9a7-4a932ef659d461cf5666bb07;729f72cd-62a4-406f-aadf-236ce7143aaa) | outputs/campaigns/security-llm-pilot/logs/baseline__llama3_2.log |

## Методология и воспроизводимость

Все параметры взяты из `configs/runtime.yaml`. Кампания запускается командой `python scripts/lab.py all`, сохраняет resume-state в `outputs/campaigns/security-llm-pilot/state.json` и отдельный лог каждой стадии рядом. Baseline, pre-train и RL checkpoint оцениваются одним evaluator и одним фиксированным срезом repository eval + checked-in external benchmark.

## Ограничения валидности

- Это pilot-профиль с ограниченным числом шагов и примеров; он проверяет весь контур, но не доказывает сходимость или превосходство метода.
- Текущие `grpo`/`gspo` — внутренние KL-регуляризованные policy-optimization baselines; отдельная reward model и rollout-group training пока не реализованы.
- Для single-GPU full-policy alignment frozen reference загружается в NF4; это memory/точность компромисс, тогда как PEFT-ветки используют BF16 reference.
- Простые lexical/actionability метрики не заменяют экспертную оценку качества security-ответов.
- Для публикационного результата нужны несколько seeds, confidence intervals, ablations, лицензионный manifest датасетов и значительно больший evaluation split.

## Артефакты

- Campaign state: `outputs/campaigns/security-llm-pilot/state.json`.
- Runtime config: `configs/runtime.yaml`.
- Stage JSON и manifests: `outputs/experiments/`.
- TensorBoard: `logs/<model>/<run_id>/tensorboard/`.
- MLflow: URI из `configs/runtime.yaml`.
