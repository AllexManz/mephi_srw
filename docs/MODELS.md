# Базовые модели

Поддерживаемые профили находятся в `configs/model/`:

| ID | Base model |
|---|---|
| `gpt2` | `gpt2` |
| `mistral` | `mistralai/Mistral-7B-v0.1` |
| `qwen2` | `Qwen/Qwen2-1.5B-Instruct` |
| `qwen2_5` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `phi3` | `microsoft/Phi-3-mini-4k-instruct` |
| `gemma2` | `google/gemma-2-2b-it` |
| `llama3_2` | `meta-llama/Llama-3.2-3B-Instruct` |
| `deepseek_r1_1_5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |

Добавление модели требует нового YAML-профиля с `name`, `id`, dtype, device map и batch settings. PEFT-методы используют автоматический fallback для attention-модулей, поэтому профиль не должен вручную перечислять одинаковые имена для всех архитектур.

`trust_remote_code` отключён по умолчанию. Включать его можно только для проверенной модели и осознанно.

Базовая модель, локальный full checkpoint и PEFT adapter — разные сущности. Их пути фиксируются в `outputs/experiments/<experiment>/manifest.json`.
