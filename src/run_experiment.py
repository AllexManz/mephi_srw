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
        f"training.training.max_steps={args.max_steps}",
    ]
    if init_adapter:
        command.append(f"peft.peft.init_adapter_path={init_adapter}")
    if pretrained_path:
        command.append(f"model.model.pretrained_path={pretrained_path}")
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
        "--torch-dtype", args.torch_dtype,
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

        _save_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="security-llm")
    parser.add_argument("--models", default="gpt2")
    parser.add_argument("--phase", choices=["all", "baseline", "pretrain", "rl", "evaluate"], default="all")
    parser.add_argument("--pretrain-method", choices=["lora", "qlora", "prompt_tuning", "p_tuning", "full"], default="lora")
    parser.add_argument("--rl-method", choices=["gspo", "grpo", "none"], default="gspo")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--benchmark-limit", type=int, default=30)
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    ns = parser.parse_args()
    ns.models = [item.strip() for item in ns.models.split(",") if item.strip()]
    ns.model_names = {
        "gpt2": "gpt2",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "qwen2": "Qwen/Qwen2-1.5B-Instruct",
        "qwen2_5": "Qwen/Qwen2.5-1.5B-Instruct",
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
        "gemma2": "google/gemma-2-2b-it",
        "llama3_2": "meta-llama/Llama-3.2-3B-Instruct",
        "deepseek_r1_1_5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    }
    result = run(ns)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
