# Security Language Model Assistant

Проект для обучения и оценки языковых моделей на задачах информационной безопасности. Он объединяет подготовку CVE/MITRE и SIEM-данных, SFT/LoRA/QLoRA, экспериментальные GSPO/GRPO, оценку security-метрик и отдельный набор тестов prompt injection, data leakage и data poisoning.

## Состав проекта

- `src/data/` — загрузка и подготовка CVE, MITRE ATT&CK, CWE, NIST и OWASP;
- `src/training/` — датасет, Trainer, PEFT и policy-optimization;
- `src/evaluation/` — perplexity, accuracy, security-метрики и внешние benchmark-ы;
- `security_testing/` — генераторы атак, защиты, mock smoke-тесты и отчёты;
- `airflow/dags/` — DAG с кубиками prepare → baseline → pretrain → RL → evaluation;
- `notebooks/` — просмотр данных, генераций и сравнение стадий;
- `docs/` — отдельная документация моделей, данных, pretrain, RL, benchmark-ов, Airflow и MLflow;
- `configs/` — Hydra-конфигурации моделей, обучения, PEFT, датасетов и логирования;
- `data/processed/` — локальные train/eval-данные;
- `models/`, `logs/`, `outputs/` — результаты запусков (большие артефакты не предназначены для git).

## Установка

Требуется Python 3.9+ и, для большинства моделей, CUDA GPU. `setup.sh` создаёт conda-окружение, но для CUDA-версии PyTorch используйте команду с [официальной страницы PyTorch](https://pytorch.org/get-started/locally/).

```bash
conda create -n security_llm python=3.9 -y
conda activate security_llm
python -m pip install -r requirements.txt
```

Для воспроизводимой установки reference-окружения:

```bash
python -m pip install -r requirements.lock
```

Скопируйте настройки:

```bash
cp .env.example .env
```

В `.env` укажите `NVD_API_KEY`, если нужен NVD API, и замените локальные MinIO credentials. Файл `.env` не коммитится.

## Подготовка датасета

```bash
python src/data/prepare_dataset.py dataset=small
```

Доступны профили `small`, `full` и `base`. Скрипт использует сетевые источники с timeout 30 секунд и создаёт непустые `train.json` и `eval.json`. Если источники недоступны, используются локальные fallback-примеры; при недостатке данных запуск завершается с понятной ошибкой.

## Обучение и оценка

Полный experiment pipeline для нескольких моделей:

```bash
python src/run_experiment.py \
  --experiment security-llm \
  --models gpt2,qwen2_5 \
  --pretrain-method lora \
  --rl-method gspo \
  --benchmark-limit 30
```

Для проверки команд без загрузки моделей:

```bash
python src/run_experiment.py --models gpt2 --dry-run
```

Полный контракт и stage artifacts описаны в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/PRETRAIN.md](docs/PRETRAIN.md) и [docs/RL.md](docs/RL.md).

Примеры:

```bash
python src/train.py model=gpt2 peft=lora
python src/train.py model=mistral peft=qlora
python src/train.py model=qwen2_5 peft=lora
python src/evaluate.py model=gpt2 peft=lora
```

Доступные конфигурации моделей: `gpt2`, `mistral`, `qwen2`, `qwen2_5`, `phi3`, `gemma2`, `llama3_2`, `deepseek_r1_1_5b`. Большинство моделей требуют доступа к Hugging Face и значительных ресурсов. PEFT-режимы: `lora`, `qlora`, `prefix_tuning`, `prompt_tuning`, `base`.

Пути и параметры можно переопределять через Hydra:

```bash
python src/train.py model=gpt2 peft=lora \
  model.model.training.max_length=256 \
  model.model.training.per_device_train_batch_size=1
```

FSDP запускайте через `torchrun` и подходящий preset:

```bash
torchrun --nproc_per_node=2 src/train.py model=gpt2 training=fsdp_gpt2
```

TensorBoard:

```bash
tensorboard --logdir logs
```

## SIEM и MLflow

Обычный finetuning через `src/train.py` автоматически создаёт MLflow run, записывает параметры конфигурации, train/eval metrics на каждом шаге и итоговый adapter/checkpoint как artifact. Включение находится в `configs/logging/base.yaml`.

Локальный MLflow UI:

```bash
python -m pip install -r requirements.txt
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

В отдельном терминале запустите обучение:

```bash
MLFLOW_TRACKING_URI=file:./mlruns \
python src/train.py model=gpt2 peft=lora
```

Откройте `http://127.0.0.1:5000`, выберите experiment `security-llm-finetuning` и нужный run. Для удалённого MLflow достаточно заменить `MLFLOW_TRACKING_URI`, например на `http://localhost:5000`.

Расширенная локальная инструкция: [docs/MLFLOW.md](docs/MLFLOW.md).

Простой SIEM-запуск:

```bash
python src/run_siem_simple.py
python src/run_siem.py --mode monitor --interval 60 --time-window 5
```

Полный pipeline: Elasticsearch → датасет → QLoRA → GSPO → evaluation:

```bash
python src/train_pipeline.py
```

Локальный MLflow можно запустить без Docker:

```bash
export MLFLOW_TRACKING_URI=file:./mlruns
python src/train_pipeline.py
```

Для Docker сначала заполните `.env`, затем:

```bash
docker compose up -d
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/train_pipeline.py
```

Compose намеренно не содержит credentials по умолчанию и завершится с ошибкой, если `MINIO_ROOT_USER` или `MINIO_ROOT_PASSWORD` не заданы.

## Security testing

Быстрый mock-прогон без весов модели:

```bash
PYTHONPATH=. python3 security_testing/evaluation/run_full_evaluation.py --limit 4
```

Генерация тестовых наборов и реальные прогоны описаны в `security_testing/evaluation/` и используют `security_testing/configs/model_config.yaml`. Загрузка `trust_remote_code` по умолчанию отключена: включайте её только для проверенного репозитория, явно изменив конфигурацию и осознавая риск выполнения стороннего Python-кода.

## Тесты

Быстрые тесты не скачивают модели:

```bash
python run_tests.py
```

Модельные snapshot-тесты — интеграционные, требуют сети, Hugging Face cache, CPU/GPU и могут занимать много времени. Запускайте их отдельно:

```bash
RUN_MODEL_TESTS=1 python run_tests.py
```

Результаты оценки сохраняются в `logs/`, адаптеры и checkpoints — в `models/`. Эти директории могут занимать десятки гигабайт и не должны попадать в обычные коммиты.

## Метрики

Основные метрики: perplexity, accuracy, average reward, threat detection accuracy, false-positive/false-negative rate, MITRE ATT&CK coverage, sequential event handling, response time, multi-stage detection и temporal context accuracy. Метрика сохраняется как `null`, если в eval-примерах нет необходимых полей.

## Безопасность и воспроизводимость

- Не храните API keys, MLflow credentials и модельные токены в git.
- Не включайте `trust_remote_code` для непроверенных моделей.
- Используйте `requirements.lock` для reference-окружения; обновляйте lock-файл осознанно при смене стека.
- Не используйте production secrets в локальном `docker-compose`.

## Лицензия

MIT
