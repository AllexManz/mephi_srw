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

2. Создайте и настройте окружение:
```bash
# Создание conda окружения и установка зависимостей
bash setup.sh

# Активация окружения
conda activate security_llm

# Создание .env файла из примера
cp .env.example .env
```

3. Настройте переменные окружения в `.env`:
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

1. Соберите датасет:
```bash
python src/data/collect_cve.py  # Сбор данных CVE
python src/data/collect_mitre.py  # Сбор данных MITRE ATT&CK
```

2. Обработайте данные:
```bash
python src/data/process_dataset.py
```

### Обучение

1. Базовое обучение с Mistral-7B и LoRA:
```bash
python src/train.py model=mistral peft=lora
```

2. Обучение с GPT-2 (для тестирования):
```bash
python src/train.py model=gpt2 peft=lora
```

3. Дополнительные опции:
```bash
# Изменение параметров обучения
python src/train.py model=mistral peft=lora training.training.num_train_epochs=5

# Отключение LoRA
python src/train.py model=mistral peft=none

# Изменение размера батча
python src/train.py model=mistral peft=lora training.training.per_device_train_batch_size=2
```

### Мониторинг обучения

1. TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

2. Проверка логов:
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

- `configs/config.yaml` - основная конфигурация
- `configs/model/` - конфигурации моделей
- `configs/peft/` - конфигурации методов PEFT
- `configs/training/` - параметры обучения

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