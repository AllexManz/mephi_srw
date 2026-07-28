# Архитектура pipeline

```text
data preparation
      |
      v
baseline benchmark ---> MLflow run + artifacts
      |
      v
pre-train (PEFT/full) ---> TensorBoard + MLflow + checkpoint
      |
      v
RL alignment (GRPO/GSPO) ---> TensorBoard + MLflow + checkpoint
      |
      v
stage benchmarks ---> JSON report + MLflow metrics + notebooks
```

Пользовательский entrypoint: `scripts/lab.py`; конфигурационный контракт:
`configs/runtime.yaml`. Низкоуровневый CLI `src/run_experiment.py` сохраняет
manifest в `outputs/experiments/<name>/manifest.json`.

`src/run_campaign.py` добавляет resume-state, обработку частичных ошибок,
автоматический выбор лучшей baseline-модели и cross-product пяти pre-train ×
двух alignment-методов. State хранится в
`outputs/campaigns/<name>/state.json`.

Airflow вызывает те же CLI-команды по отдельности. Это делает pipeline воспроизводимым локально и в scheduler.
