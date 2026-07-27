# Локальная документация по MLflow

Этот проект использует MLflow для отслеживания finetuning-запусков. Обычный запуск через `src/train.py` создаёт run, записывает параметры и train/eval-метрики, а итоговые checkpoints и adapters сохраняет в artifacts.

## Быстрый запуск

Установить зависимости:

```bash
python -m pip install -r requirements.txt
```

Запустить локальный MLflow UI в отдельном терминале:

```bash
mlflow ui \
  --backend-store-uri ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

Открыть в браузере:

```text
http://127.0.0.1:5000
```

В другом терминале запустить finetuning:

```bash
MLFLOW_TRACKING_URI=file:./mlruns \
python src/train.py model=gpt2 peft=lora
```

Experiment по умолчанию называется `security-llm-finetuning`.

## Что логируется

Для каждого запуска MLflow получает:

- параметры модели и базовой модели;
- метод обучения: SFT, GSPO или GRPO;
- PEFT-метод: LoRA, QLoRA, prefix/prompt tuning или full fine-tuning;
- размеры train/eval-датасетов;
- Python-версию, git SHA и tracking URI;
- полную Hydra-конфигурацию как `config/config.yaml`;
- train metrics на шагах Trainer: `loss`, `learning_rate`, `epoch` и другие scalar-значения;
- eval metrics: `eval_loss`, `eval_perplexity` и связанные значения;
- итоговые checkpoints, adapters и tokenizer в artifact-папке `model/`.

Метрики можно смотреть во вкладке `Metrics`, параметры — во вкладке `Parameters`, а конфигурацию и модельные файлы — во вкладке `Artifacts`.

## Конфигурация

Основная конфигурация находится в [configs/logging/base.yaml](../configs/logging/base.yaml):

```yaml
mlflow:
  enabled: true
  tracking_uri: file:./mlruns
  experiment_name: security-llm-finetuning
  run_name: ${now:%Y-%m-%d_%H-%M-%S}
```

MLflow можно отключить для локального запуска без tracking:

```bash
python src/train.py \
  model=gpt2 peft=lora \
  logging.mlflow.enabled=false
```

При отключённом MLflow TensorBoard и остальные включённые логгеры продолжают работать.

## Tracking URI

Переменная окружения `MLFLOW_TRACKING_URI` имеет приоритет над значением в Hydra-конфигурации.

Локальный файловый backend:

```bash
export MLFLOW_TRACKING_URI=file:./mlruns
```

Удалённый MLflow server:

```bash
export MLFLOW_TRACKING_URI=http://mlflow.example.local:5000
python src/train.py model=gpt2 peft=lora
```

Если сервер требует авторизацию, передайте её способом, принятым вашей инфраструктурой, например через переменные окружения или reverse proxy. Не записывайте credentials в YAML и git.

## Docker Compose

В репозитории есть Compose-конфигурация с MLflow и MinIO. Перед запуском создайте `.env`:

```bash
cp .env.example .env
```

Задайте в `.env` непредсказуемые значения:

```dotenv
MINIO_ROOT_USER=local-mlflow-user
MINIO_ROOT_PASSWORD=replace-with-a-long-password
```

Запустите сервисы:

```bash
docker compose up -d
docker compose ps
```

MLflow UI будет доступен по адресу `http://127.0.0.1:5000`. Для запуска обучения с хост-машины:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/train.py model=gpt2 peft=lora
```

Остановить сервисы:

```bash
docker compose down
```

Удаление volume с артефактами является разрушительной операцией — не используйте `docker compose down -v`, если хотите сохранить данные MLflow/MinIO.

## Pipeline с SIEM

Для полного pipeline используется `src/train_pipeline.py`. Он открывает отдельный MLflow run и дополнительно логирует:

- количество событий SIEM;
- train/eval-датасеты;
- конфигурацию pipeline;
- adapters после stage 1 и stage 2;
- evaluation metrics.

Локальный запуск:

```bash
MLFLOW_TRACKING_URI=file:./mlruns \
python src/train_pipeline.py
```

Конфигурация pipeline находится в [configs/pipeline/base.yaml](../configs/pipeline/base.yaml).

## Полезные команды

Показать список experiments:

```bash
mlflow experiments search
```

Посмотреть runs через CLI:

```bash
mlflow runs list --experiment-name security-llm-finetuning
```

Проверить, что локальный tracking store существует:

```bash
find mlruns -maxdepth 3 -type f | sort | head -50
```

Посмотреть размер artifacts:

```bash
du -sh mlruns
```

## Типовые проблемы

### `ModuleNotFoundError: No module named 'mlflow'`

Установите зависимости:

```bash
python -m pip install -r requirements.txt
```

Либо временно отключите интеграцию:

```bash
python src/train.py logging.mlflow.enabled=false
```

### Run не появляется в UI

Проверьте, что UI и обучение используют один и тот же tracking URI:

```bash
echo "$MLFLOW_TRACKING_URI"
```

Для файлового backend оба процесса должны работать с одним путём `./mlruns` и из одного корня проекта.

### Run создан, но метрики отсутствуют

Проверьте:

1. что `logging.mlflow.enabled=true`;
2. что обучение не завершилось до создания Trainer;
3. что в UI открыт правильный experiment;
4. что в логе есть шаги Trainer и `logging_steps` не слишком велик;
5. что не используется другой `MLFLOW_TRACKING_URI` из окружения.

### Artifacts занимают слишком много места

Trainer может сохранять несколько checkpoint-ов. Настройки находятся в `configs/training/base.yaml`:

```yaml
training:
  save:
    save_total_limit: 3
```

Уменьшите `save_total_limit`, отключите частое сохранение или удалите старые runs через MLflow API после проверки, что они больше не нужны.

## Безопасность

- Не коммитьте `.env`, API keys, tokens и пароли.
- Не используйте production credentials в локальном Compose.
- Ограничивайте доступ к удалённому MLflow UI.
- Помните, что artifacts могут содержать веса моделей и данные обучения.
- Перед отправкой artifacts на внешний сервер проверьте датасеты на наличие секретов и персональных данных.
