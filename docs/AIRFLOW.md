# Airflow

Airflow предоставляет UI и scheduler для стадий pipeline. DAG находится в
`airflow/dags/security_llm_pipeline.py` и при парсинге читает только
`configs/runtime.yaml`.

```text
prepare_data -> baseline_benchmarks -> pretrain -> rl_alignment -> post_training_benchmarks
```

## Запуск

```bash
python scripts/lab.py bootstrap
python scripts/lab.py up
python scripts/lab.py status
```

UI по умолчанию: `http://127.0.0.1:8080`. Airflow home, DAG folder, host, port и
флаг example DAG задаются в секции `services.airflow` runtime-конфига.
Переменные `SECURITY_LLM_*` больше не поддерживаются.

Launcher запускает `airflow standalone` в изолированном
`.runtime/airflow-venv`. Внутренние Airflow bootstrapping variables создаются
только для дочернего процесса из YAML; пользователь не управляет ими вручную.
Лог сервиса: `.runtime/services/logs/airflow.log`.

DAG вызывает тот же `src/run_experiment.py`, что CLI и notebooks. Airflow
хранит orchestration state; MLflow хранит параметры, метрики и artifacts.
Долгая полная сравнительная кампания дополнительно доступна напрямую через
`python scripts/lab.py all`, поскольку resume-state и обработка частичных
ошибок там надежнее одного монолитного DAG run.
