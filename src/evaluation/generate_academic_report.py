"""Generate a paper-style Markdown report from a campaign state file."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime_config import ROOT, load_runtime_config


QUALITY_METRICS = ("accuracy", "safety", "word_overlap", "actionability")


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(value: str | Path) -> str:
    """Prefer portable repository-relative paths in the committed report."""
    path = Path(value)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _result_metrics(entry: Dict[str, Any]) -> Dict[str, Any]:
    path = entry.get("result")
    if not path or not Path(path).exists():
        return {}
    return _load(Path(path)).get("metrics", {})


def _environment() -> Dict[str, str]:
    details = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("torch", "transformers", "peft"):
        try:
            details[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            details[package] = "not-installed"
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            details["gpu"] = f"{props.name} ({props.total_memory / 1024**3:.1f} GiB)"
        else:
            details["gpu"] = "CUDA unavailable"
    except ImportError:
        details["gpu"] = "torch unavailable"
    return details


def _quality_score(metrics: Dict[str, Any]) -> Any:
    values = [float(metrics[key]) for key in QUALITY_METRICS if key in metrics]
    return sum(values) / len(values) if values else None


def _effective_status(entry: Dict[str, Any]) -> str:
    status = entry.get("status", "not-started")
    path = entry.get("result")
    if status == "completed" and (not path or not Path(path).exists()):
        return "missing-artifact"
    return status


def _diagnostic(entry: Dict[str, Any], log_path: Path) -> str:
    if entry.get("error"):
        return str(entry["error"]).replace("|", "\\|")
    if not log_path.exists():
        return "см. лог"
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    markers = (
        "GatedRepoError",
        "403 Client Error",
        "CUDA out of memory",
        "Repository Not Found",
        "Could not override",
        "ValueError:",
        "RuntimeError:",
        "CalledProcessError:",
    )
    for marker in markers:
        matches = [line for line in lines if marker in line]
        if matches:
            return matches[-1][:240].replace("|", "\\|")
    return "процесс завершился ненулевым кодом"


def _table(rows: Iterable[List[str]], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def generate(config_path: str, state_path: Path, output: Path) -> Path:
    runtime = load_runtime_config(config_path)
    state = _load(state_path)
    campaign = runtime["campaign"]
    training = campaign["training"]
    benchmark = campaign["benchmark"]
    environment = _environment()
    declared_hardware = campaign.get("hardware", {})
    if declared_hardware.get("accelerator"):
        environment["gpu"] = str(declared_hardware["accelerator"])

    baseline_rows = []
    baseline_scores: Dict[str, float] = {}
    metric_counts: Dict[str, Any] = {}
    for model_id in campaign["models"]:
        entry = state.get("baseline", {}).get(model_id, {})
        metrics = _result_metrics(entry)
        result_path = entry.get("result")
        if not metric_counts and result_path and Path(result_path).exists():
            metric_counts = _load(Path(result_path)).get("metric_counts", {})
        score = _quality_score(metrics)
        if _effective_status(entry) == "completed" and score is not None:
            baseline_scores[model_id] = score
        baseline_rows.append([
            model_id,
            _effective_status(entry),
            _fmt(score),
            _fmt(metrics.get("accuracy")),
            _fmt(metrics.get("safety")),
            _fmt(metrics.get("word_overlap")),
            _fmt(metrics.get("actionability")),
            _fmt(metrics.get("average_response_time_seconds")),
        ])

    baseline_winner = max(baseline_scores, key=baseline_scores.get) if baseline_scores else "не выбрана"
    winner_score = baseline_scores.get(baseline_winner)
    matrix_rows = []
    matrix_scores: Dict[str, float] = {}
    for key, entry in sorted(state.get("matrix", {}).items()):
        metrics = _result_metrics(entry)
        score = _quality_score(metrics)
        if _effective_status(entry) == "completed" and score is not None:
            matrix_scores[key] = score
        stage = "pretrain" if key.endswith("/pretrain") else "rl"
        matrix_rows.append([
            key,
            stage,
            _effective_status(entry),
            _fmt(score),
            _fmt(score - winner_score if score is not None and winner_score is not None else None),
            _fmt(metrics.get("accuracy")),
            _fmt(metrics.get("safety")),
            _fmt(metrics.get("word_overlap")),
            _fmt(metrics.get("actionability")),
            _fmt(metrics.get("average_response_time_seconds")),
        ])

    all_entries = list(state.get("baseline", {}).values()) + list(state.get("matrix", {}).values())
    completed = sum(
        _effective_status(entry) == "completed"
        for entry in all_entries
    )
    failed = sum(
        _effective_status(entry) == "failed" for entry in all_entries
    )
    best_variant = max(matrix_scores, key=matrix_scores.get) if matrix_scores else "нет завершенных"
    failed_rows = []
    for scope, entries in (("baseline", state.get("baseline", {})), ("matrix", state.get("matrix", {}))):
        for key, entry in entries.items():
            if _effective_status(entry) == "failed":
                default_log = state_path.parent / "logs" / f"{scope}__{key.replace('/', '__')}.log"
                log_path = Path(entry.get("log", default_log))
                failed_rows.append([
                    scope,
                    key,
                    _diagnostic(entry, log_path),
                    _display_path(log_path),
                ])
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        f"# {campaign['name']}: отчет об экспериментальной кампании",
        "",
        "## Аннотация",
        "",
        (
            "Отчет фиксирует воспроизводимую pilot-кампанию сравнения базовых security LLM, "
            "supervised fine-tuning и последующего policy alignment. Результаты автоматически "
            "собраны из неизменяемых JSON-артефактов; отсутствующие или упавшие прогоны не "
            "заменяются синтетическими значениями."
        ),
        "",
        "## Экспериментальная постановка",
        "",
        f"- Сгенерировано: `{generated}`.",
        f"- Лучшая по baseline модель: `{baseline_winner}`.",
        f"- Успешных стадий: `{completed}`; упавших: `{failed}`.",
        f"- Train steps: `{training['max_steps']}`; train/eval limits: `{training['train_limit']}`/`{training['eval_limit']}`.",
        f"- Benchmark limit: `{benchmark['limit']}`; max new tokens: `{benchmark['max_new_tokens']}`.",
        f"- Metric denominators: `{json.dumps(metric_counts, ensure_ascii=False, sort_keys=True)}`.",
        f"- Environment: Python `{environment['python']}`, PyTorch `{environment['torch']}`, "
        f"Transformers `{environment['transformers']}`, PEFT `{environment['peft']}`.",
        f"- Accelerator: `{environment['gpu']}`; platform: `{environment['platform']}`.",
        f"- Pre-train методы: `{', '.join(campaign['pretrain_methods'])}`.",
        f"- Alignment методы: `{', '.join(campaign['rl_methods'])}`.",
        "",
        "Главная сравнительная величина для автоматического выбора модели — среднее доступных "
        "quality-метрик accuracy, safety, word overlap и actionability. Latency в выбор не входит.",
        "",
        "## Результаты базовых моделей",
        "",
        _table(
            baseline_rows,
            ["Model", "Status", "Score", "Accuracy", "Safety", "Overlap", "Actionability", "Latency, s"],
        ),
        "",
        "## Результаты обучения и alignment",
        "",
        _table(
            matrix_rows,
            ["Variant", "Stage", "Status", "Score", "Δ baseline", "Accuracy", "Safety", "Overlap", "Actionability", "Latency, s"],
        ) if matrix_rows else "Матрица обучения еще не запускалась.",
        "",
        "## Основные наблюдения",
        "",
        f"- Победитель baseline: `{baseline_winner}` со score `{_fmt(winner_score)}`.",
        f"- Лучший завершенный обученный вариант: `{best_variant}` со score `{_fmt(matrix_scores.get(best_variant))}`.",
        "- Разности в таблице относятся к baseline выбранной модели; из-за pilot-размера их нельзя трактовать как статистически значимый эффект.",
        "",
        "## Неуспешные стадии",
        "",
        _table(failed_rows, ["Scope", "Variant", "Reason", "Diagnostic log"])
        if failed_rows
        else "Неуспешных стадий нет.",
        "",
        "## Методология и воспроизводимость",
        "",
        "Все параметры взяты из `configs/runtime.yaml`. Кампания запускается командой "
        "`python scripts/lab.py all`, сохраняет resume-state в "
        f"`outputs/campaigns/{campaign['name']}/state.json` и отдельный лог каждой стадии рядом. "
        "Baseline, pre-train и RL checkpoint оцениваются одним evaluator и одним фиксированным "
        "срезом repository eval + checked-in external benchmark.",
        "",
        "## Ограничения валидности",
        "",
        "- Это pilot-профиль с ограниченным числом шагов и примеров; он проверяет весь контур, "
        "но не доказывает сходимость или превосходство метода.",
        "- Текущие `grpo`/`gspo` — внутренние KL-регуляризованные policy-optimization baselines; "
        "отдельная reward model и rollout-group training пока не реализованы.",
        "- Для single-GPU full-policy alignment frozen reference загружается в NF4; "
        "это memory/точность компромисс, тогда как PEFT-ветки используют BF16 reference.",
        "- Простые lexical/actionability метрики не заменяют экспертную оценку качества security-ответов.",
        "- Для публикационного результата нужны несколько seeds, confidence intervals, ablations, "
        "лицензионный manifest датасетов и значительно больший evaluation split.",
        "",
        "## Артефакты",
        "",
        f"- Campaign state: `{_display_path(state_path)}`.",
        f"- Runtime config: `{_display_path(runtime['_config_path'])}`.",
        "- Stage JSON и manifests: `outputs/experiments/`.",
        "- TensorBoard: `logs/<model>/<run_id>/tensorboard/`.",
        "- MLflow: URI из `configs/runtime.yaml`.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--state")
    parser.add_argument("--output", default="reports/EXPERIMENT_REPORT.md")
    args = parser.parse_args()
    runtime = load_runtime_config(args.config)
    state = Path(args.state) if args.state else (
        ROOT / "outputs" / "campaigns" / runtime["campaign"]["name"] / "state.json"
    )
    print(generate(args.config, state, Path(args.output)))


if __name__ == "__main__":
    main()
