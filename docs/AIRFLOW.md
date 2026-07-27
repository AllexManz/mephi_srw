# Airflow

Airflow используется как UI и scheduler для кубиков pipeline. DAG: `airflow/dags/security_llm_pipeline.py`.

Граф:

```text
prepare_data -> baseline_benchmarks -> pretrain -> rl_alignment -> post_training_benchmarks
```

Настройки задаются переменными окружения Airflow:

```bash
export SECURITY_LLM_REPO=/home/papkovas/mephi_srw
export SECURITY_LLM_MODELS=gpt2,qwen2_5
export SECURITY_LLM_PRETRAIN=lora
export SECURITY_LLM_RL=gspo
export SECURITY_LLM_BENCHMARK_LIMIT=30
```

Установка Airflow в отдельное окружение:

```bash
python -m pip install -r requirements-airflow.txt
export AIRFLOW_HOME=$PWD/.airflow
airflow db migrate
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.local --password change-me
airflow scheduler
airflow webserver --port 8080
```

Проектные ML-зависимости должны быть установлены в том же окружении либо доступны через `SECURITY_LLM_PYTHON`. Для первого smoke-прогона используйте `--dry-run` у `src/run_experiment.py`.

Airflow хранит orchestration state, а MLflow — параметры, метрики и artifacts. Это разные ответственности; Airflow не заменяет MLflow.
