# MLflow

MLflow отслеживает finetuning и benchmark-запуски: конфигурацию, git SHA,
параметры, train/eval metrics и artifacts модели. TensorBoard events пишутся
параллельно и доступны отдельным UI.

## Запуск без переменных окружения

```bash
python scripts/lab.py bootstrap
python scripts/lab.py up
python scripts/lab.py status
```

MLflow UI по умолчанию: `http://127.0.0.1:5000`. Server работает в
`.runtime/mlflow-venv`, а training использует совместимый lightweight client.
SQLite metadata хранится в
`.runtime/mlflow/mlflow.db`, artifacts — в `.runtime/mlflow/artifacts`.
Tracking URI, experiment names, host и port задаются в `configs/runtime.yaml`.

`src/train.py` получает URI через явный Hydra key
`logging.mlflow.tracking_uri`; `src/run_experiment.py` передает туда значение
runtime-конфига. Значения окружения больше не переопределяют этот контракт.
В training requirements закреплен `mlflow-skinny`: полный пакет MLflow 2.19
требует `pyarrow<19`, тогда как `datasets 4.8` требует `pyarrow>=21`. Полный
server поэтому устанавливается отдельно командой bootstrap.

## Что логируется

- model id/name, PEFT и training method;
- train/eval split sizes, Python version и git SHA;
- полная Hydra-конфигурация;
- loss, learning rate, epoch и scalar Trainer metrics;
- eval loss/perplexity, когда evaluation включена;
- adapter/full checkpoint и tokenizer;
- benchmark metrics, число примеров для каждой метрики и JSON с samples.

## Прямой запуск

Для автономного `src/train.py` default остается файловым и задается в
`configs/logging/base.yaml`:

```bash
python src/train.py model=gpt2 peft=lora
```

Чтобы направить такой запуск в поднятый server, используйте явный override:

```bash
python src/train.py model=gpt2 peft=lora \
  logging.mlflow.tracking_uri=http://127.0.0.1:5000
```

## Опциональный Docker/MinIO backend

Основной локальный путь не требует Docker. Если доступен Docker Compose:

```bash
python scripts/lab.py init
docker compose up -d
```

Compose читает два Docker secret-файла из `.secrets/`, а не `.env`. Никогда не
коммитьте `.secrets`; `docker compose down -v` удалит volume с артефактами.

## Диагностика

```bash
python scripts/lab.py doctor
python scripts/lab.py status
tail -f .runtime/services/logs/mlflow.log
```

Если run отсутствует, сравните URI в resolved Hydra config с
`services.mlflow.tracking_uri`. Если модуль не установлен, выполните
`python scripts/lab.py bootstrap`. При переполнении диска уменьшите
`save_total_limit`, pilot limits или удалите только заранее проверенные старые
артефакты.
