# Запуск и эксплуатация

Единая точка настройки — `configs/runtime.yaml`, единая точка запуска —
`scripts/lab.py`. Пользователю не нужно экспортировать `MLFLOW_TRACKING_URI`,
`AIRFLOW_HOME`, параметры моделей или MinIO credentials.

## Быстрый путь

```bash
python scripts/lab.py doctor
python scripts/lab.py all --dry-run --best-model gpt2
python scripts/lab.py all
```

`all` выполняет `init → bootstrap (только при отсутствии venv) → doctor → up →
baseline → matrix → report`. Повторный запуск безопасно продолжает кампанию из
`outputs/campaigns/<campaign>/state.json`.

`up` идемпотентен: перед запуском каждого UI он проверяет настроенный порт и
повторно использует уже работающий MLflow, Airflow или TensorBoard. Поэтому
повторная команда не создает дублирующие webserver/scheduler-процессы.

## Команды

| Команда | Назначение |
|---|---|
| `python scripts/lab.py init` | создать `.runtime`, `.secrets` и каталоги артефактов |
| `python scripts/lab.py bootstrap` | подготовить раздельные training/MLflow и Airflow venv |
| `python scripts/lab.py doctor --require-gpu` | проверить зависимости, CUDA и свободный диск |
| `python scripts/lab.py up` | запустить три UI-сервиса в фоне |
| `python scripts/lab.py status` | показать PID и адреса UI |
| `python scripts/lab.py down` | остановить только процессы, запущенные launcher-ом |
| `python scripts/lab.py baseline` | оценить все модели из runtime-конфига |
| `python scripts/lab.py matrix --best-model gpt2` | принудительно выбрать модель для матрицы |
| `python scripts/lab.py report` | пересобрать `reports/EXPERIMENT_REPORT.md` |

Сервисные PID и логи лежат в `.runtime/services/`. Training logs и TensorBoard
events — в `logs/<model>/<run_id>/`; веса — в `models/<model>/checkpoints/`.
Bootstrap использует официальный `https://pypi.org/simple` и официальный
Airflow constraints-файл. Training venv получает только `mlflow-skinny` client
и использует системный CUDA-стек; полный MLflow server и Airflow находятся в
двух отдельных venv. Такое разделение исключает конфликт `datasets/pyarrow` и
Flask/protobuf. Для первого запуска нужен сетевой доступ.

## Надежность конфигурации

Настройки сервисов, матрицы, лимиты и model registry находятся в одном YAML.
CLI-флаги имеют явный приоритет только для текущего запуска. Код больше не
читает `MLFLOW_TRACKING_URI` или `SECURITY_LLM_*` из окружения.
Фактически использованный аппаратный профиль pilot-кампании также зафиксирован
в `campaign.hardware` и переносится в автоматически генерируемый отчёт.

Airflow как продукт требует bootstrapping-параметры через process environment.
Launcher формирует их внутри дочернего процесса строго из `runtime.yaml`; их не
нужно экспортировать, и они не влияют на shell пользователя.

Локальные MinIO credentials генерируются один раз криптографически стойким
генератором и хранятся с правами `0600` в `.secrets/`. Они нужны только
опциональному `docker-compose.yml`; основной локальный MLflow хранит metadata в
SQLite, artifacts — в `.runtime/mlflow/artifacts`.

## Профили эксперимента

Checked-in значения — pilot-профиль: 10 steps, 256 train, 64 eval и 30
benchmark-примеров. Он предназначен для проверки полного контура и сравнения
на фиксированном малом бюджете. Для полноценного исследования увеличьте эти
значения в `configs/runtime.yaml`, заранее проверив GPU и диск.

Reference-профиль оставляет `dataloader_num_workers: 0`: на текущем kernel 5.4
это устраняет риск зависания fork/tokenizers. На kernel 5.5+ значение можно
увеличить после отдельного smoke-теста.

Campaign runner сохраняет один явный финальный artifact каждой стадии и
отключает внутренний last-step checkpoint Trainer. Это не дублирует full model
и optimizer state; для длинных production-запусков с resume включите
`save_strategy` в отдельном training-профиле.

Если gated-модель недоступна или отдельный вариант падает, при
`continue_on_error: true` ошибка фиксируется в state/log, а кампания продолжает
остальные независимые варианты. Отчет не подставляет mock-результаты.
Если training artifact уже сохранён, но упал последующий benchmark, resume
повторяет только benchmark и не переобучает модель.
