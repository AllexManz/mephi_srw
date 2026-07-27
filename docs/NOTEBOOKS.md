# Notebooks

Ноутбуки находятся в `notebooks/`:

- `01_data_overview.ipynb` — размеры, поля, источники и примеры данных;
- `02_generation_samples.ipynb` — samples генераций из benchmark result JSON;
- `03_stage_comparison.ipynb` — сравнение baseline/pretrain/RL по MLflow/JSON метрикам.

Запуск:

```bash
jupyter lab notebooks
```

Ноутбуки не должны запускать полное обучение по умолчанию. Тяжёлые операции должны быть оформлены как явные параметры `LIMIT`, `MODEL_ID` и `RUN_ID`.
