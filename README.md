# Security Language Model Assistant

Проект по дообучению языковой модели для задач информационной безопасности с использованием PEFT (Parameter-Efficient Fine-Tuning) методов.

## Структура проекта

```
.
├── configs/              # Конфигурационные файлы
│   ├── model/           # Конфигурации моделей (Mistral, GPT-2)
│   ├── peft/            # Конфигурации методов PEFT (LoRA)
│   └── training/        # Конфигурации обучения
├── data/                # Директория для датасетов
│   ├── raw/            # Исходные данные
│   └── processed/      # Обработанные данные для обучения
├── logs/               # Директория для логов
│   ├── tensorboard/    # Логи TensorBoard
│   └── training/       # Логи обучения
├── models/             # Директория для моделей
│   ├── adapters/       # LoRA адаптеры
│   ├── checkpoints/    # Чекпоинты моделей
│   └── tokenizer/      # Токенизаторы
├── notebooks/          # Jupyter notebooks для анализа
├── src/                # Исходный код
│   ├── data/          # Скрипты для обработки данных
│   └── train.py       # Основной скрипт обучения
├── security_testing/  # Инструментарий безопасностных экспериментов (этап 4)
├── .env.example        # Пример файла с переменными окружения
├── requirements.txt    # Зависимости проекта
└── setup.sh           # Скрипт для настройки окружения
```

## Установка и настройка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/your-username/security-llm.git
cd security-llm
```

1. Создайте и настройте окружение:

```bash
# Создание conda окружения и установка зависимостей
bash setup.sh

# Активация окружения
conda activate security_llm

# Создание .env файла из примера
cp .env.example .env
```

1. Настройте переменные окружения в `.env`:

```bash
# Пути к директориям (опционально)
BASE_DIR=/path/to/project
MODELS_DIR=/path/to/models
LOGS_DIR=/path/to/logs
CACHE_DIR=/path/to/cache

# API ключи (обязательно для сбора данных)
NVD_API_KEY=your_nvd_api_key
MITRE_ATTACK_URL=https://attack.mitre.org/enterprise/attack.json
```

## Обучение модели

### Подготовка данных

1. Подготовьте датасет:

```bash
python src/data/prepare_dataset.py dataset=full
```

### Обучение

1. Базовое обучение с Mistral-7B и LoRA:

```bash
python src/train.py model=mistral peft=lora
```

1. Обучение с GPT-2 (для тестирования):

```bash
python src/train.py model=gpt2 peft=lora
```

1. Дополнительные опции:

```bash
# Изменение параметров обучения
python src/train.py model=mistral peft=lora training.training.num_train_epochs=5

# Отключение LoRA
python src/train.py model=mistral peft=none

# Изменение размера батча
python src/train.py model=mistral peft=lora training.training.per_device_train_batch_size=2
```

### RL стадия (GSPO/GRPO)

RL‑стадия обучает модель с KL‑регуляризацией относительно референс‑модели.

```bash
python src/train.py training.training.method=gspo
python src/train.py training.training.method=grpo
```

Параметры RL‑стадии находятся в `configs/training/base.yaml`:

- `training.policy_optimization.kl_coef` — вес KL‑регуляризации
- `training.policy_optimization.temperature` — температура в KL‑терме

### Мониторинг обучения

1. TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

1. Проверка логов:

```bash
# Логи обучения
ls logs/training/

# Чекпоинты модели
ls models/checkpoints/

# LoRA адаптеры
ls models/adapters/
```

## Конфигурация

Проект использует Hydra для управления конфигурацией. Основные конфигурационные файлы:

- `configs/base.yaml` - основная конфигурация
- `configs/model/` - конфигурации моделей
- `configs/peft/` - конфигурации методов PEFT
- `configs/training/` - параметры обучения

## Интеграция с SIEM

Минимальный запуск анализа последних событий:

```bash
python src/run_siem_simple.py
```

Режим мониторинга/исторического анализа:

```bash
python src/run_siem.py --mode monitor --interval 60 --time-window 5
python src/run_siem.py --mode analyze --start-time 2025-01-01T00:00:00
```

## FSDP обучение

Запуск под FSDP лучше делать через `torchrun`:

```bash
torchrun --nproc_per_node=2 src/train.py training=fsdp_gpt2
```

Для Mistral:

```bash
torchrun --nproc_per_node=2 src/train.py training=fsdp_mistral model=mistral
```

## Пайплайн обучения

Полный пайплайн: сбор событий из SIEM → обработка → QLoRA → GSPO → оценка.

```bash
python src/train_pipeline.py
```

## Метрики оценки

Считаются и логируются в `evaluation_metrics.json` и TensorBoard:

- Average Reward
- Threat Detection Accuracy
- False Positive Rate
- False Negative Rate
- Expert Preference Rate
- MITRE ATT&CK Coverage
- Sequential Event Handling
- Average Response Time
- Multi-stage Attack Detection
- Temporal Context Accuracy

### Как считаются метрики

- Average Reward — среднее по полю `reward` (или числу, извлеченному из ответа).
- Threat Detection Accuracy — доля совпадений бинарного класса «угроза»/«нет угрозы» по `threat_level` (high/medium → угроза, low → нет).
- False Positive Rate — FP / N, где N — число негативных примеров.
- False Negative Rate — FN / P, где P — число позитивных примеров.
- Expert Preference Rate — доля точных совпадений с `expert_preferred_output`.
- MITRE ATT&CK Coverage — доля найденных ID (TXXXX[.XXX]) от ожидаемого списка `mitre_attack`.
- Sequential Event Handling — доля примеров, где в ответе упомянут `sequence_index`.
- Average Response Time — среднее `response_time_seconds` (из генерации).
- Multi-stage Attack Detection — доля примеров, где в ответе присутствует хотя бы один из `attack_stages`.
- Temporal Context Accuracy — доля примеров, где в ответе присутствует `temporal_label`.

Если поле отсутствует — метрика не считается и сохраняется как `null`.

### Требуемые поля в eval.json

- `reward`
- `threat_level` или `is_threat`
- `expert_preferred_output`
- `mitre_attack` (List[str])
- `sequence_id`, `sequence_index`
- `attack_stages` (List[str])
- `temporal_label`

## Эксперименты по безопасности (`security_testing/`)

Артефакты генерации: `security_testing/test_cases/`; результаты прогонов: `security_testing/results/` (локальные JSON).

Из корня репозитория задайте `PYTHONPATH=.`, затем выполните сценарий:

```bash
# 1. Датасеты
PYTHONPATH=. python3 security_testing/datasets/generate_prompt_injection.py
PYTHONPATH=. python3 security_testing/datasets/generate_data_leakage.py
PYTHONPATH=. python3 security_testing/datasets/generate_poisoned_data.py

# 2. Быстрый smoke-тест без весов модели
PYTHONPATH=. python3 security_testing/evaluation/run_full_evaluation.py --limit 4

# 3. Реальный Mistral + QLoRA (GPU и адаптер по `security_testing/configs/model_config.yaml`)
PYTHONPATH=. python3 security_testing/attacks/run_prompt_injection.py --output security_testing/results/pi_baseline.json
PYTHONPATH=. python3 security_testing/attacks/run_prompt_injection.py --with-defense --output security_testing/results/pi_protected.json
PYTHONPATH=. python3 security_testing/attacks/run_data_leakage.py --output security_testing/results/dl_baseline.json
PYTHONPATH=. python3 security_testing/attacks/run_poisoning.py --output security_testing/results/poison_baseline.json

# 4. Сводка для отчёта
PYTHONPATH=. python3 security_testing/evaluation/compare_results.py \
  --pi-base security_testing/results/pi_baseline.json \
  --pi-sec security_testing/results/pi_protected.json \
  --dl-base security_testing/results/dl_baseline.json \
  --dl-sec security_testing/results/dl_protected.json \
  --poison-base security_testing/results/poison_baseline.json \
  --poison-def security_testing/results/poison_protected.json \
  --out security_testing/results/final_comparison.json

PYTHONPATH=. python3 security_testing/evaluation/generate_tables.py \
  --pi-base security_testing/results/pi_baseline.json \
  --pi-sec security_testing/results/pi_protected.json \
  --poison security_testing/results/poison_protected.json \
  --out security_testing/tables/report.md
```

Дополнительно: `security_testing/configs/defense_config.yaml`. DP‑SGD без жёсткой привязки к `train.py` — см. `security_testing/defenses/dp_training.py` (заглушка + опционально `opacus`).

## MLflow

По умолчанию используется локальное хранилище (`file:./mlruns`). Можно указать внешний сервер:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/train_pipeline.py
```

Для полноценной интеграции с внешним артефакт-хранилищем используйте `docker-compose`:

```bash
docker compose up -d
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/train_pipeline.py
```

## Требования

### Минимальные требования

- Python 3.9+
- 16GB RAM
- 50GB свободного места на диске

### Рекомендуемые требования для обучения

- CUDA-совместимая GPU с 16GB+ VRAM
- 32GB+ RAM
- 100GB+ свободного места на диске

### Рекомендуемые требования для инференса

- CUDA-совместимая GPU с 8GB+ VRAM
- 16GB+ RAM
- 20GB+ свободного места на диске

## Лицензия

MIT

## Примечания

- Для обучения Mistral-7B требуется GPU с достаточным объемом VRAM
- Для тестирования можно использовать GPT-2, который требует меньше ресурсов
- Все пути к директориям можно настроить через переменные окружения
- Логи и чекпоинты сохраняются в соответствующих директориях, структура которых сохраняется в git

