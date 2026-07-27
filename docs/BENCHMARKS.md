# Benchmarks

Оценка должна выполняться для трёх состояний каждой модели:

1. `baseline` — исходные веса;
2. `pretrain` — checkpoint после выбранного PEFT/full метода;
3. `rl` — checkpoint после GRPO/GSPO.

`src/evaluation/run_stage_benchmark.py` объединяет:

- локальный `data/processed/eval.json`;
- внешний checked-in subset в `data/external/external_benchmark_1500.json`, собранный из CyberMetric и CyberSecEval-derived задач.

Метрики включают exact/MCQ accuracy, prompt-injection safety, word overlap, actionability и latency. Для production-исследования нужно хранить версии источников, лицензии и дату выгрузки вместе с dataset manifest.

Запуск одного stage:

```bash
python src/evaluation/run_stage_benchmark.py \
  --model-id gpt2 --model gpt2 --stage baseline \
  --limit 30 --output outputs/benchmarks/gpt2_baseline.json
```

Сравнительный pipeline запускается через `src/run_experiment.py`.
