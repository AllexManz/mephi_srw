"""
Training pipeline with SIEM ingestion, two-stage training, and evaluation.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

import hydra
import torch
from elasticsearch import Elasticsearch
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import mlflow
    from mlflow import transformers as mlflow_transformers
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None
    mlflow_transformers = None

from evaluation.evaluator import SecurityModelEvaluator
from training.dataset import SecurityDataset
from training.trainer import SecurityModelTrainer
from training.utils import update_paths_with_run_id
from training.mlflow_logging import is_active as mlflow_active


def _get_git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return None


def _log_text_artifact(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    if mlflow_active():
        mlflow.log_artifact(str(path))


def _init_mlflow(cfg: DictConfig) -> Optional[Any]:
    mlflow_cfg = cfg.pipeline.mlflow
    if not mlflow_cfg.enabled:
        return None
    if mlflow is None:
        raise ImportError("mlflow is not installed. Add it to requirements.txt.")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_cfg.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.experiment_name)
    run = mlflow.start_run(run_name=mlflow_cfg.run_name)

    mlflow.set_tag("pipeline", "siem-qlora-gspo-eval")
    git_sha = _get_git_sha()
    if git_sha:
        mlflow.set_tag("git_sha", git_sha)
    mlflow.log_param("python_version", sys.version.split()[0])
    mlflow.log_param("tracking_uri", tracking_uri)

    if mlflow_transformers:
        mlflow_transformers.autolog()
    return run


def _fetch_siem_events(cfg: DictConfig) -> List[Dict[str, Any]]:
    es_cfg = cfg.elasticsearch
    client = Elasticsearch(
        hosts=es_cfg.hosts,
        basic_auth=(es_cfg.username, es_cfg.password) if es_cfg.username and es_cfg.password else None,
        verify_certs=es_cfg.verify_certs
    )
    time_window = cfg.pipeline.siem.time_window_minutes
    max_events = cfg.pipeline.siem.max_events
    query = cfg.pipeline.siem.query

    search_query = {
        "size": max_events,
        "sort": [{es_cfg.time_field: "desc"}],
        "query": {
            "bool": {
                "must": [
                    {"match_all": {}}
                ]
            }
        }
    }
    if time_window:
        from datetime import datetime, timedelta
        time_threshold = datetime.utcnow() - timedelta(minutes=time_window)
        search_query["query"]["bool"]["must"].append({
            "range": {
                es_cfg.time_field: {"gte": time_threshold.isoformat()}
            }
        })
    if query:
        search_query["query"]["bool"]["must"].extend(query.get("must", []))
        search_query["query"]["bool"]["should"] = query.get("should", [])
        search_query["query"]["bool"]["filter"] = query.get("filter", [])

    response = client.search(index=es_cfg.index_pattern, body=search_query)
    events = []
    for hit in response["hits"]["hits"]:
        event = hit["_source"]
        event["_id"] = hit["_id"]
        event["_score"] = hit["_score"]
        events.append(event)
    return events


def _process_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = []
    for event in events:
        threat = event.get("threat_level") or event.get("severity") or "низкий"
        analysis = event.get("analysis") or event.get("message") or ""
        output = f"Уровень угрозы: {threat}."
        if analysis:
            output = f"{output}\nАнализ: {analysis}"

        examples.append({
            "instruction": "Проанализируй событие безопасности и определи уровень угрозы.",
            "input": json.dumps(event, ensure_ascii=False, indent=2),
            "output": output,
            "context": None,
            "threat_level": threat
        })
    return examples


def _save_dataset(examples: List[Dict[str, Any]], cfg: DictConfig) -> None:
    output_dir = Path(cfg.pipeline.dataset.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.pipeline.dataset.seed)
    random.shuffle(examples)

    split_idx = int(len(examples) * cfg.pipeline.dataset.train_val_split_ratio)
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]

    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_examples, f, ensure_ascii=False, indent=2)
    with open(output_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_examples, f, ensure_ascii=False, indent=2)

    if mlflow_active():
        mlflow.log_artifact(str(output_dir / "train.json"))
        mlflow.log_artifact(str(output_dir / "eval.json"))


def _log_config_artifacts(cfg: DictConfig) -> None:
    if not mlflow_active():
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        _log_text_artifact(tmp_path / "config.yaml", OmegaConf.to_yaml(cfg))
        _log_text_artifact(tmp_path / "config.json", json.dumps(
            OmegaConf.to_container(cfg, resolve=True),
            ensure_ascii=False,
            indent=2
        ))


def _load_peft_config(cfg: DictConfig, name: str) -> DictConfig:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "peft" / f"{name}.yaml"
    return OmegaConf.load(config_path)


def _stage_train(cfg: DictConfig, peft_cfg_name: str, training_method: str, adapters_dir: str) -> str:
    stage_cfg = OmegaConf.merge(cfg, _load_peft_config(cfg, peft_cfg_name))
    stage_cfg.training.training.method = training_method
    stage_cfg.paths.adapters_dir = adapters_dir

    trainer = SecurityModelTrainer(
        model_name=stage_cfg.model.model.name,
        output_dir=stage_cfg.paths.checkpoints_dir,
        cfg=stage_cfg
    )

    dataset_path = Path(stage_cfg.paths.dataset_dir)
    with open(dataset_path / "train.json", "r", encoding="utf-8") as f:
        train_examples = json.load(f)
    with open(dataset_path / "eval.json", "r", encoding="utf-8") as f:
        eval_examples = json.load(f)

    train_dataset = SecurityDataset(train_examples, trainer.tokenizer, stage_cfg.model.model.training.max_length)
    eval_dataset = SecurityDataset(eval_examples, trainer.tokenizer, stage_cfg.model.model.training.max_length)
    datasets = {"train": train_dataset, "validation": eval_dataset}

    trainer.train(datasets)

    adapter_path = os.path.join(adapters_dir, f"{stage_cfg.peft.peft.method}_adapter")
    if mlflow_active() and Path(adapter_path).exists():
        mlflow.log_artifact(adapter_path)
    return adapter_path


def _stage_gspo(cfg: DictConfig, peft_cfg_name: str, adapters_dir: str, adapter_path: str) -> None:
    stage_cfg = OmegaConf.merge(cfg, _load_peft_config(cfg, peft_cfg_name))
    stage_cfg.training.training.method = "gspo"
    stage_cfg.paths.adapters_dir = adapters_dir
    stage_cfg.peft.peft.adapter_path = adapter_path

    trainer = SecurityModelTrainer(
        model_name=stage_cfg.model.model.name,
        output_dir=stage_cfg.paths.checkpoints_dir,
        cfg=stage_cfg
    )

    # Load adapter weights from stage 1
    trainer.model = PeftModel.from_pretrained(trainer.model, adapter_path)

    dataset_path = Path(stage_cfg.paths.dataset_dir)
    with open(dataset_path / "train.json", "r", encoding="utf-8") as f:
        train_examples = json.load(f)
    with open(dataset_path / "eval.json", "r", encoding="utf-8") as f:
        eval_examples = json.load(f)

    train_dataset = SecurityDataset(train_examples, trainer.tokenizer, stage_cfg.model.model.training.max_length)
    eval_dataset = SecurityDataset(eval_examples, trainer.tokenizer, stage_cfg.model.model.training.max_length)
    datasets = {"train": train_dataset, "validation": eval_dataset}

    trainer.train(datasets)

    gspo_adapter = os.path.join(adapters_dir, f"{stage_cfg.peft.peft.method}_adapter")
    if mlflow_active() and Path(gspo_adapter).exists():
        mlflow.log_artifact(gspo_adapter)


def _evaluate(cfg: DictConfig, adapter_path: str) -> Dict[str, Any]:
    model = None
    tokenizer_source = cfg.paths.tokenizer_dir if Path(cfg.paths.tokenizer_dir).exists() else cfg.model.model.name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.pad_token = tokenizer.eos_token

    if cfg.peft.peft.enabled:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model.name,
            torch_dtype=getattr(torch, cfg.model.model.torch_dtype),
            device_map=cfg.model.model.device_map,
            trust_remote_code=cfg.model.model.trust_remote_code
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model.name,
            torch_dtype=getattr(torch, cfg.model.model.torch_dtype),
            device_map=cfg.model.model.device_map,
            trust_remote_code=cfg.model.model.trust_remote_code
        )

    eval_dataset_path = Path(cfg.paths.dataset_dir) / "eval.json"
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_examples = json.load(f)

    eval_dataset = SecurityDataset(eval_examples, tokenizer, cfg.model.model.training.max_length)
    evaluator = SecurityModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        output_dir=cfg.paths.evaluation_dir
    )
    metrics = evaluator.evaluate(eval_dataset)
    responses = evaluator.generate_responses(eval_examples)
    metrics.update(evaluator.evaluate_security_metrics(eval_examples, responses))
    evaluator.save_metrics(metrics)
    evaluator.close()
    if mlflow_active():
        metrics_path = Path(cfg.paths.evaluation_dir) / "evaluation_metrics.json"
        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path))
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="pipeline/base")
def main(cfg: DictConfig) -> None:
    # Генерируем Run ID и динамически обновляем пути
    run_id = update_paths_with_run_id(cfg)
    print(f"Generated Run ID: {run_id}")
    print(OmegaConf.to_yaml(cfg))

    if cfg.training.training.get("fsdp", {}).get("enabled", False):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < 2:
            print("Warning: FSDP enabled but WORLD_SIZE=1. Use torchrun for multi-GPU.")

    run = _init_mlflow(cfg)

    try:
        _log_config_artifacts(cfg)
        events = _fetch_siem_events(cfg)
        examples = _process_events(events)
        _save_dataset(examples, cfg)

        if mlflow_active() and run:
            mlflow.log_param("siem_events", len(events))
            mlflow.log_param("stage1_peft", cfg.pipeline.stage1.peft_config)
            mlflow.log_param("stage2_peft", cfg.pipeline.stage2.peft_config)
            mlflow.log_param("stage2_method", cfg.pipeline.stage2.training_method)

        stage1_adapter = _stage_train(
            cfg,
            cfg.pipeline.stage1.peft_config,
            cfg.pipeline.stage1.training_method,
            cfg.pipeline.stage1.adapters_dir
        )

        stage2_adapter_dir = cfg.pipeline.stage2.adapters_dir
        _stage_gspo(
            cfg,
            cfg.pipeline.stage2.peft_config,
            stage2_adapter_dir,
            stage1_adapter
        )

        eval_cfg = OmegaConf.merge(cfg, _load_peft_config(cfg, cfg.pipeline.stage2.peft_config))
        stage2_adapter_path = os.path.join(
            stage2_adapter_dir,
            f"{eval_cfg.peft.peft.method}_adapter"
        )
        metrics = _evaluate(eval_cfg, stage2_adapter_path)
        if mlflow_active() and run:
            for key, value in metrics.items():
                if value is not None:
                    mlflow.log_metric(key, float(value))
    finally:
        if mlflow_active() and run:
            mlflow.end_run()


if __name__ == "__main__":
    main()
