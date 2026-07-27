# Данные

Источники проекта:

- `data/processed/train.json` и `eval.json` — instruction dataset проекта;
- `data/external/` — локальные benchmark subsets;
- `security_testing/test_cases/` — prompt injection, leakage и poisoning cases.

Подготовка:

```bash
python src/data/prepare_dataset.py dataset=small
python src/data/prepare_dataset.py dataset=full
```

Перед обучением проверяйте:

- непустые train/eval splits;
- отсутствие секретов и персональных данных;
- баланс источников и классов угроз;
- совместимость полей `instruction`, `input`, `output`, `context`.

Dataset files и benchmark manifests должны логироваться в MLflow как artifacts, но крупные локальные данные не следует без необходимости коммитить в git.
