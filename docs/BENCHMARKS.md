# Benchmarks

Оценка должна выполняться для трёх состояний каждой модели:

1. `baseline` — исходные веса;
2. `pretrain` — checkpoint после выбранного PEFT/full метода;
3. `rl` — checkpoint после GRPO/GSPO.

`src/evaluation/run_stage_benchmark.py` объединяет:

- локальный `data/processed/eval.json`;
- внешний checked-in subset в `data/external/external_benchmark_1500.json`, собранный из CyberMetric и CyberSecEval-derived задач.

Метрики включают exact/MCQ accuracy, prompt-injection safety, word overlap,
actionability и latency. Каждая quality-метрика усредняется только по примерам,
для которых она определена; denominator сохраняется в `metric_counts`.

Запуск одного stage:

```bash
python src/evaluation/run_stage_benchmark.py \
  --model-id gpt2 --model gpt2 --stage baseline \
  --limit 30 --output outputs/benchmarks/gpt2_baseline.json
```

Сравнительный pipeline запускается через `src/run_experiment.py`.
При ручном повторе можно ограничить оценку конкретным артефактом через
`--evaluate-stage pretrain` или `--evaluate-stage rl`; campaign runner делает
это автоматически и не пересчитывает уже измеренную стадию.

Полная кампания и A*-style Markdown report:

```bash
python scripts/lab.py baseline
python scripts/lab.py matrix
python scripts/lab.py report
```

Отчет в `reports/EXPERIMENT_REPORT.md` следует академической структуре, но не
скрывает ограничений pilot-профиля. Для публикационного результата нужны
несколько seeds, confidence intervals, ablations, экспертная оценка и dataset
license/version manifest.
