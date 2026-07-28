"""Load and validate the repository runtime configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "runtime.yaml"
PRETRAIN_METHODS = {"lora", "qlora", "prompt_tuning", "p_tuning", "full"}
RL_METHODS = {"grpo", "gspo"}


class RuntimeConfigError(ValueError):
    """Raised when the runtime configuration is incomplete or inconsistent."""


def _require(mapping: Dict[str, Any], keys: Iterable[str], section: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise RuntimeConfigError(
            f"Missing keys in runtime config section '{section}': {', '.join(missing)}"
        )


def resolve_path(value: str | Path, root: Path = ROOT) -> Path:
    """Resolve a runtime path relative to the repository root."""
    path = Path(value).expanduser()
    # Do not follow venv/bin/python symlinks back to /usr/bin/python.
    return path.absolute() if path.is_absolute() else (root / path).absolute()


def load_runtime_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load runtime YAML and return a validated plain dictionary."""
    config_path = resolve_path(path or DEFAULT_CONFIG)
    if not config_path.exists():
        raise FileNotFoundError(f"Runtime config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RuntimeConfigError("Runtime config root must be a mapping")

    _require(data, ("project", "services", "campaign", "models"), "root")
    _require(
        data["project"],
        (
            "root",
            "bootstrap_python",
            "python",
            "mlflow_python",
            "airflow_python",
            "runtime_dir",
        ),
        "project",
    )
    _require(data["services"], ("mlflow", "airflow", "tensorboard"), "services")
    _require(
        data["services"]["mlflow"],
        (
            "enabled",
            "host",
            "port",
            "tracking_uri",
            "backend_store",
            "artifact_root",
            "experiment_name",
            "benchmark_experiment_name",
        ),
        "services.mlflow",
    )
    _require(
        data["services"]["airflow"],
        ("enabled", "host", "port", "home", "dags_folder", "load_examples"),
        "services.airflow",
    )
    _require(
        data["services"]["tensorboard"],
        ("enabled", "host", "port", "log_dir"),
        "services.tensorboard",
    )
    _require(
        data["campaign"],
        ("name", "models", "pretrain_methods", "rl_methods", "training", "benchmark"),
        "campaign",
    )
    _require(
        data["campaign"]["training"],
        (
            "max_steps",
            "train_limit",
            "eval_limit",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
        ),
        "campaign.training",
    )
    _require(
        data["campaign"]["benchmark"],
        ("limit", "max_length", "max_new_tokens", "samples_in_report"),
        "campaign.benchmark",
    )

    unknown_models = sorted(set(data["campaign"]["models"]) - set(data["models"]))
    if unknown_models:
        raise RuntimeConfigError(
            "Campaign references unknown model ids: " + ", ".join(unknown_models)
        )
    unknown_pretrain = sorted(
        set(data["campaign"]["pretrain_methods"]) - PRETRAIN_METHODS
    )
    if unknown_pretrain:
        raise RuntimeConfigError(
            "Campaign references unknown pretrain methods: "
            + ", ".join(unknown_pretrain)
        )
    unknown_rl = sorted(set(data["campaign"]["rl_methods"]) - RL_METHODS)
    if unknown_rl:
        raise RuntimeConfigError(
            "Campaign references unknown RL methods: " + ", ".join(unknown_rl)
        )
    for service_name, service in data["services"].items():
        port = service.get("port")
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise RuntimeConfigError(
                f"Invalid TCP port in services.{service_name}: {port!r}"
            )
    for model_id, model in data["models"].items():
        _require(model, ("name", "torch_dtype"), f"models.{model_id}")

    data["_config_path"] = str(config_path)
    data["_root"] = str(ROOT)
    return data
