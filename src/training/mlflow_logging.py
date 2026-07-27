"""Small, optional MLflow integration shared by training entry points."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

try:
    import mlflow
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    mlflow = None


def is_active() -> bool:
    """Return whether MLflow is installed and a run is currently active."""
    return mlflow is not None and mlflow.active_run() is not None


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def start_run(
    cfg: DictConfig,
    run_id: str,
    *,
    train_examples: Optional[int] = None,
    eval_examples: Optional[int] = None,
) -> Any:
    """Start the configured run and log stable metadata/configuration."""
    mlflow_cfg = cfg.logging.mlflow
    if not mlflow_cfg.enabled:
        return None
    if mlflow is None:
        raise ImportError(
            "MLflow logging is enabled, but mlflow is not installed. "
            "Install requirements.txt or set logging.mlflow.enabled=false."
        )

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", str(mlflow_cfg.tracking_uri)
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(mlflow_cfg.experiment_name))
    run = mlflow.start_run(
        run_name=str(mlflow_cfg.run_name),
        tags={
            "entrypoint": "src/train.py",
            "run_id": run_id,
            "git_sha": _git_sha() or "unknown",
        },
    )

    params = {
        "model_name": str(cfg.model.model.name),
        "model_id": str(cfg.model.model.id),
        "training_method": str(cfg.training.training.method),
        "peft_method": (
            str(cfg.peft.peft.method) if cfg.peft.peft.enabled else "full"
        ),
        "python_version": sys.version.split()[0],
        "tracking_uri": tracking_uri,
    }
    if train_examples is not None:
        params["train_examples"] = train_examples
    if eval_examples is not None:
        params["eval_examples"] = eval_examples
    mlflow.log_params(params)
    mlflow.log_text(OmegaConf.to_yaml(cfg), "config/config.yaml")
    return run


def log_artifacts(path: str, artifact_path: str) -> None:
    if is_active():
        mlflow.log_artifacts(path, artifact_path=artifact_path)


def end_run(run: Any, *, status: Optional[str] = None) -> None:
    if mlflow is not None and run is not None and mlflow.active_run() is not None:
        mlflow.end_run(status=status)


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log finite scalar metrics and ignore Trainer metadata MLflow cannot store."""
    if not is_active():
        return
    clean: Dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric == numeric and abs(numeric) != float("inf"):
                clean[name.replace("/", "_")] = numeric
    if clean:
        mlflow.log_metrics(clean, step=step)
