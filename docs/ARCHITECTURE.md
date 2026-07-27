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

Главный CLI: `src/run_experiment.py`. Он сохраняет manifest в `outputs/experiments/<name>/manifest.json`. Этот manifest связывает базовую модель, pretrain artifact, RL artifact и benchmark outputs.

Airflow вызывает те же CLI-команды по отдельности. Это делает pipeline воспроизводимым локально и в scheduler.
