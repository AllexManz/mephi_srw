"""Run a resumable baseline and pretrain/RL experiment campaign."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from runtime_config import ROOT, load_runtime_config


QUALITY_METRICS = ("accuracy", "safety", "word_overlap", "actionability")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _quality_score(result: Dict[str, Any]) -> float:
    metrics = result.get("metrics", {})
    values = [float(metrics[key]) for key in QUALITY_METRICS if key in metrics]
    return sum(values) / len(values) if values else float("-inf")


class CampaignRunner:
    def __init__(self, config_path: str, dry_run: bool = False):
        self.config_path = str(Path(config_path))
        self.runtime = load_runtime_config(config_path)
        self.campaign = self.runtime["campaign"]
        self.name = self.campaign["name"]
        self.dry_run = dry_run
        self.output_dir = ROOT / "outputs" / "campaigns" / self.name
        self.state_path = self.output_dir / "state.json"
        self.log_dir = self.output_dir / "logs"
        if dry_run:
            self.state = {
                "campaign": self.name,
                "config": str(Path(config_path)),
                "created_at": _now(),
                "baseline": {},
                "matrix": {},
            }
        elif self.state_path.exists():
            self.state = _read_json(self.state_path)
        else:
            self.state = {
                "campaign": self.name,
                "config": str(Path(config_path)),
                "created_at": _now(),
                "baseline": {},
                "matrix": {},
            }
        self.state["updated_at"] = _now()
        self._save()

    def _save(self) -> None:
        if self.dry_run:
            return
        self.state["updated_at"] = _now()
        _write_json(self.state_path, self.state)

    def _run(self, key: str, command: List[str]) -> bool:
        printable = " ".join(command)
        print(f"\n[{key}] $ {printable}", flush=True)
        if self.dry_run:
            return True

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{key.replace('/', '__')}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {printable}\n")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return process.wait() == 0

    def _log_path(self, key: str) -> str:
        return str(self.log_dir / f"{key.replace('/', '__')}.log")

    def _experiment_command(
        self,
        experiment: str,
        model_id: str,
        phase: str,
        *extra: str,
    ) -> List[str]:
        return [
            sys.executable,
            "src/run_experiment.py",
            "--config",
            self.config_path,
            "--experiment",
            experiment,
            "--models",
            model_id,
            "--phase",
            phase,
            *extra,
        ]

    def baseline(self) -> None:
        experiment = f"{self.name}-baseline"
        for model_id in self.campaign["models"]:
            entry = self.state["baseline"].setdefault(model_id, {})
            result_path = ROOT / "outputs" / "experiments" / experiment / f"{model_id}_baseline.json"
            if entry.get("status") == "completed" and result_path.exists():
                print(f"[baseline/{model_id}] already completed; resume skips it")
                continue
            entry.clear()
            entry.update(
                status="running",
                started_at=_now(),
                result=str(result_path),
                log=self._log_path(f"baseline/{model_id}"),
            )
            self._save()
            model_cfg = self.runtime["models"][model_id]
            ok = self._run(
                f"baseline/{model_id}",
                self._experiment_command(
                    experiment,
                    model_id,
                    "baseline",
                    "--torch-dtype",
                    model_cfg["torch_dtype"],
                ),
            )
            if ok and (self.dry_run or result_path.exists()):
                entry.update(status="completed", finished_at=_now())
                if result_path.exists():
                    entry["score"] = _quality_score(_read_json(result_path))
            else:
                entry.update(status="failed", finished_at=_now())
            self._save()
            if not ok and not self.campaign.get("continue_on_error", True):
                raise RuntimeError(f"Baseline failed for {model_id}")

    def select_best_model(self, requested: Optional[str] = None) -> str:
        if requested:
            if requested not in self.runtime["models"]:
                raise ValueError(f"Unknown requested model: {requested}")
            selected = requested
        else:
            candidates = {
                model: entry.get("score", float("-inf"))
                for model, entry in self.state.get("baseline", {}).items()
                if entry.get("status") == "completed"
            }
            if not candidates:
                if self.dry_run:
                    selected = self.campaign["models"][0]
                else:
                    raise RuntimeError("No successful baseline result is available")
            else:
                selected = max(candidates, key=candidates.get)
        self.state["selected_model"] = selected
        self._save()
        print(f"Selected model for training matrix: {selected}")
        return selected

    def _copy_pretrain_manifest(
        self, source_experiment: str, target_experiment: str, model_id: str
    ) -> None:
        source = ROOT / "outputs" / "experiments" / source_experiment / "manifest.json"
        target = ROOT / "outputs" / "experiments" / target_experiment / "manifest.json"
        if self.dry_run:
            return
        source_manifest = _read_json(source)
        pretrain = source_manifest["models"][model_id]["pretrain"]
        _write_json(
            target,
            {
                "experiment": target_experiment,
                "models": {model_id: {"pretrain": pretrain}},
            },
        )

    def _pretrain_artifact_exists(self, experiment: str, model_id: str) -> bool:
        manifest_path = ROOT / "outputs" / "experiments" / experiment / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            pretrain = _read_json(manifest_path)["models"][model_id]["pretrain"]
        except (KeyError, TypeError, json.JSONDecodeError):
            return False
        artifact = pretrain.get("adapter_path") or pretrain.get("model_path")
        return bool(artifact and Path(artifact).is_dir())

    def matrix(self, model_id: Optional[str] = None) -> None:
        model_id = self.select_best_model(model_id or self.state.get("selected_model"))
        for pretrain_method in self.campaign["pretrain_methods"]:
            pretrain_key = f"{pretrain_method}/pretrain"
            pretrain_entry = self.state["matrix"].setdefault(pretrain_key, {})
            pretrain_experiment = f"{self.name}-{model_id}-{pretrain_method}"
            pretrain_result = (
                ROOT / "outputs" / "experiments" / pretrain_experiment / f"{model_id}_pretrain.json"
            )
            if not (
                pretrain_entry.get("status") == "completed" and pretrain_result.exists()
            ):
                pretrain_entry.clear()
                pretrain_entry.update(
                    status="running",
                    started_at=_now(),
                    result=str(pretrain_result),
                    log=self._log_path(pretrain_key),
                )
                self._save()
                if not self.dry_run and self._pretrain_artifact_exists(
                    pretrain_experiment, model_id
                ):
                    print(f"[{pretrain_key}] reusing existing training artifact")
                    train_ok = True
                else:
                    train_ok = self._run(
                        pretrain_key,
                        self._experiment_command(
                            pretrain_experiment,
                            model_id,
                            "pretrain",
                            "--pretrain-method",
                            pretrain_method,
                        ),
                    )
                eval_ok = train_ok and self._run(
                    f"{pretrain_method}/evaluate",
                    self._experiment_command(
                        pretrain_experiment,
                        model_id,
                        "evaluate",
                        "--evaluate-stage",
                        "pretrain",
                    ),
                )
                if eval_ok and (self.dry_run or pretrain_result.exists()):
                    pretrain_entry.update(status="completed", finished_at=_now())
                    if pretrain_result.exists():
                        pretrain_entry["score"] = _quality_score(_read_json(pretrain_result))
                else:
                    pretrain_entry.update(status="failed", finished_at=_now())
                self._save()

            if pretrain_entry.get("status") != "completed" and not self.dry_run:
                if not self.campaign.get("continue_on_error", True):
                    raise RuntimeError(f"Pretrain failed: {pretrain_method}")
                continue

            for rl_method in self.campaign["rl_methods"]:
                key = f"{pretrain_method}/{rl_method}"
                entry = self.state["matrix"].setdefault(key, {})
                pair_experiment = f"{pretrain_experiment}-{rl_method}"
                result_path = (
                    ROOT / "outputs" / "experiments" / pair_experiment / f"{model_id}_rl.json"
                )
                if entry.get("status") == "completed" and result_path.exists():
                    print(f"[{key}] already completed; resume skips it")
                    continue
                entry.clear()
                entry.update(
                    status="running",
                    started_at=_now(),
                    result=str(result_path),
                    log=self._log_path(key),
                )
                self._save()
                try:
                    self._copy_pretrain_manifest(
                        pretrain_experiment, pair_experiment, model_id
                    )
                    train_ok = self._run(
                        key,
                        self._experiment_command(
                            pair_experiment,
                            model_id,
                            "rl",
                            "--pretrain-method",
                            pretrain_method,
                            "--rl-method",
                            rl_method,
                        ),
                    )
                    eval_ok = train_ok and self._run(
                        f"{key}/evaluate",
                        self._experiment_command(
                            pair_experiment,
                            model_id,
                            "evaluate",
                            "--evaluate-stage",
                            "rl",
                        ),
                    )
                except Exception:
                    traceback.print_exc()
                    eval_ok = False
                if eval_ok and (self.dry_run or result_path.exists()):
                    entry.update(status="completed", finished_at=_now())
                    if result_path.exists():
                        entry["score"] = _quality_score(_read_json(result_path))
                else:
                    entry.update(status="failed", finished_at=_now())
                self._save()
                if not eval_ok and not self.campaign.get("continue_on_error", True):
                    raise RuntimeError(f"RL stage failed: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--phase", choices=("baseline", "matrix", "all"), default="all")
    parser.add_argument("--best-model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runner = CampaignRunner(args.config, args.dry_run)
    if args.phase in ("baseline", "all"):
        runner.baseline()
    if args.phase in ("matrix", "all"):
        runner.matrix(args.best_model)


if __name__ == "__main__":
    main()
