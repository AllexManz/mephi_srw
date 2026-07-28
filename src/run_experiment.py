"""Orchestrate baseline -> pretrain -> RL alignment -> benchmarks.

The runner is intentionally a thin process orchestrator: each training and
benchmark stage remains independently runnable and creates its own MLflow run.
This makes the same stages usable from a shell, Airflow or a notebook.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from runtime_config import load_runtime_config


ROOT = Path(__file__).resolve().parents[1]


def _run_id(experiment: str, model_id: str, stage: str) -> str:
    return f"{experiment}_{model_id}_{stage}_{uuid.uuid4().hex[:8]}"


def _command(args: List[str], dry_run: bool) -> None:
    printable = " ".join(str(item) for item in args)
    print(f"\n$ {printable}")
    if not dry_run:
        subprocess.run(args, cwd=ROOT, check=True)


def _train_command(
    model_id: str,
    peft_method: str,
    run_id: str,
    args: argparse.Namespace,
    *,
    init_adapter: Optional[str] = None,
    pretrained_path: Optional[str] = None,
) -> List[str]:
    peft_config = "base" if peft_method == "full" else peft_method
    command = [
        sys.executable,
        "src/train.py",
        f"model={model_id}",
        f"peft={peft_config}",
        f"paths.run_id={run_id}",
        "logging.mlflow.enabled=true",
        f"logging.mlflow.tracking_uri={args.tracking_uri}",
        f"logging.mlflow.experiment_name={args.mlflow_experiment}",
        f"training.training.max_steps={args.max_steps}",
        f"training.training.data.train_limit={args.train_limit}",
        f"training.training.data.eval_limit={args.eval_limit}",
        f"model.model.training.per_device_train_batch_size={args.train_batch_size}",
        f"model.model.training.per_device_eval_batch_size={args.eval_batch_size}",
        f"model.model.training.gradient_accumulation_steps={args.gradient_accumulation_steps}",
        # Campaign stages save one explicit final artifact. Disabling Trainer's
        # last-step checkpoint avoids duplicating full weights and optimizer
        # state before that final save.
        "training.training.save.save_strategy=no",
        "training.training.evaluation.load_best_model_at_end=false",
    ]
    if init_adapter:
        # Most PEFT config groups do not declare this transition-only field.
        # ``++`` makes the override valid both when the field exists (p-tuning)
        # and when Hydra has to add it (LoRA/QLoRA/prompt tuning).
        command.append(f"++peft.peft.init_adapter_path={init_adapter}")
    if pretrained_path:
        command.append(f"model.model.pretrained_path={pretrained_path}")
    if peft_method == "full" and args.full_optimizer:
        command.append(f"training.training.optimization.optim={args.full_optimizer}")
    if args.no_eval:
        command.append("training.training.evaluation.eval_strategy=no")
    return command


def _benchmark_command(
    model_id: str,
    model_name: str,
    stage: str,
    output: Path,
    args: argparse.Namespace,
    *,
    adapter_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> List[str]:
    command = [
        sys.executable,
        "src/evaluation/run_stage_benchmark.py",
        "--model-id", model_id,
        "--model", model_name,
        "--stage", stage,
        "--output", str(output),
        "--limit", str(args.benchmark_limit),
        "--torch-dtype", args.torch_dtype or args.model_dtypes[model_id],
        "--max-length", str(args.benchmark_max_length),
        "--max-new-tokens", str(args.max_new_tokens),
        "--samples", str(args.samples),
        "--tracking-uri", args.tracking_uri,
        "--mlflow-experiment", args.benchmark_experiment,
    ]
    if adapter_path:
        command.extend(["--adapter-path", adapter_path, "--base-model", model_name])
    if model_path:
        command.extend(["--model-path", model_path])
    return command


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"experiment": path.parent.name, "models": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_path = ROOT / "outputs" / "experiments" / args.experiment / "manifest.json"
    manifest = _load_manifest(manifest_path)
    manifest["experiment"] = args.experiment
    manifest.setdefault("models", {})

    for model_id in args.models:
        model_entry = manifest["models"].setdefault(model_id, {})
        model_name = args.model_names.get(model_id, model_id)

        if args.phase in ("all", "baseline"):
            output = manifest_path.parent / f"{model_id}_baseline.json"
            _command(
                _benchmark_command(model_id, model_name, "baseline", output, args),
                args.dry_run,
            )
            model_entry["baseline"] = {"benchmark": str(output)}

        if args.phase in ("all", "pretrain"):
            run_id = _run_id(args.experiment, model_id, "pretrain")
            _command(
                _train_command(model_id, args.pretrain_method, run_id, args),
                args.dry_run,
            )
            checkpoint_dir = ROOT / "models" / model_id / "checkpoints" / run_id
            if args.pretrain_method == "full":
                artifact = checkpoint_dir / "full_model"
                model_entry["pretrain"] = {
                    "method": "full",
                    "run_id": run_id,
                    "model_path": str(artifact),
                }
            else:
                artifact = checkpoint_dir / f"{args.pretrain_method}_adapter"
                model_entry["pretrain"] = {
                    "method": args.pretrain_method,
                    "run_id": run_id,
                    "adapter_path": str(artifact),
                }

        if args.phase in ("all", "rl") and args.rl_method != "none":
            pretrain = model_entry.get("pretrain")
            if not pretrain:
                raise RuntimeError(
                    f"No pretrain artifact for {model_id}. Run phase=pretrain first."
                )
            run_id = _run_id(args.experiment, model_id, "rl")
            init_adapter = pretrain.get("adapter_path")
            pretrained_path = pretrain.get("model_path")
            _command(
                _train_command(
                    model_id,
                    pretrain["method"],
                    run_id,
                    args,
                    init_adapter=init_adapter,
                    pretrained_path=pretrained_path,
                ) + [f"training.training.method={args.rl_method}"],
                args.dry_run,
            )
            checkpoint_dir = ROOT / "models" / model_id / "checkpoints" / run_id
            if pretrain["method"] == "full":
                model_entry["rl"] = {"run_id": run_id, "model_path": str(checkpoint_dir / "full_model")}
            else:
                model_entry["rl"] = {
                    "run_id": run_id,
                    "adapter_path": str(checkpoint_dir / f"{pretrain['method']}_adapter"),
                }

        if args.phase in ("all", "evaluate"):
            for stage in ("pretrain", "rl"):
                if args.evaluate_stage != "all" and stage != args.evaluate_stage:
                    continue
                entry = model_entry.get(stage)
                if not entry:
                    continue
                output = manifest_path.parent / f"{model_id}_{stage}.json"
                _command(
                    _benchmark_command(
                        model_id,
                        model_name,
                        stage,
                        output,
                        args,
                        adapter_path=entry.get("adapter_path"),
                        model_path=entry.get("model_path"),
                    ),
                    args.dry_run,
                )
                entry["benchmark"] = str(output)

        if not args.dry_run:
            _save_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--models")
    parser.add_argument("--phase", choices=["all", "baseline", "pretrain", "rl", "evaluate"], default="all")
    parser.add_argument(
        "--evaluate-stage",
        choices=["all", "pretrain", "rl"],
        default="all",
        help="Limit phase=evaluate to one manifest stage.",
    )
    parser.add_argument("--pretrain-method", choices=["lora", "qlora", "prompt_tuning", "p_tuning", "full"])
    parser.add_argument("--rl-method", choices=["gspo", "grpo", "none"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--benchmark-limit", type=int)
    parser.add_argument("--benchmark-max-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--torch-dtype")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    ns = parser.parse_args()
    runtime = load_runtime_config(ns.config)
    campaign = runtime["campaign"]
    training = campaign["training"]
    benchmark = campaign["benchmark"]
    mlflow = runtime["services"]["mlflow"]
    selected_models = ns.models.split(",") if ns.models else campaign["models"]
    ns.models = [item.strip() for item in selected_models if item.strip()]
    unknown = sorted(set(ns.models) - set(runtime["models"]))
    if unknown:
        parser.error("unknown model ids: " + ", ".join(unknown))
    ns.experiment = ns.experiment or campaign["name"]
    ns.pretrain_method = ns.pretrain_method or campaign["pretrain_methods"][0]
    ns.rl_method = ns.rl_method or campaign["rl_methods"][0]
    ns.max_steps = ns.max_steps if ns.max_steps is not None else training["max_steps"]
    ns.train_limit = ns.train_limit if ns.train_limit is not None else training["train_limit"]
    ns.eval_limit = ns.eval_limit if ns.eval_limit is not None else training["eval_limit"]
    ns.train_batch_size = ns.train_batch_size or training["per_device_train_batch_size"]
    ns.eval_batch_size = ns.eval_batch_size or training["per_device_eval_batch_size"]
    ns.gradient_accumulation_steps = (
        ns.gradient_accumulation_steps or training["gradient_accumulation_steps"]
    )
    ns.full_optimizer = training.get("full_optimizer")
    ns.benchmark_limit = ns.benchmark_limit or benchmark["limit"]
    ns.benchmark_max_length = ns.benchmark_max_length or benchmark["max_length"]
    ns.max_new_tokens = ns.max_new_tokens or benchmark["max_new_tokens"]
    ns.samples = ns.samples or benchmark["samples_in_report"]
    ns.tracking_uri = mlflow["tracking_uri"]
    ns.mlflow_experiment = mlflow["experiment_name"]
    ns.benchmark_experiment = mlflow["benchmark_experiment_name"]
    ns.model_names = {key: value["name"] for key, value in runtime["models"].items()}
    ns.model_dtypes = {
        key: value["torch_dtype"] for key, value in runtime["models"].items()
    }
    result = run(ns)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
