"""Modules 8 & 9 — integrity checks for adapters and regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


class ModelIntegrity:
    def __init__(self, canary_accuracy_drop: float = 0.05):
        self.canary_accuracy_drop = canary_accuracy_drop

    def hash_paths(self, files: Iterable[Path]) -> str:
        h = hashlib.sha256()
        for fp in sorted(files, key=lambda p: str(p)):
            if fp.is_file():
                h.update(fp.name.encode())
                with open(fp, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
        return h.hexdigest()

    def hash_adapter(self, adapter_dir: Path | str) -> str:
        p = Path(adapter_dir)
        if not p.is_dir():
            return ""
        binaries = []
        # LoRA payloads are *.safetensors or pytorch_model.bin
        for suffix in ("*.safetensors", "*.bin"):
            binaries.extend(sorted(p.glob(suffix)))
        if binaries:
            return self.hash_paths(binaries)
        return ""

    def should_block_deploy(self, baseline_acc: float, canary_acc: float) -> bool:
        if baseline_acc <= 0:
            return False
        return baseline_acc - canary_acc > self.canary_accuracy_drop

    def regression_flag(self, previous_metrics_path: Path, current_metrics_path: Path) -> Dict[str, Any]:
        if not previous_metrics_path.is_file() or not current_metrics_path.is_file():
            return {"flag": False, "detail": "missing_metrics"}

        prev = json.loads(previous_metrics_path.read_text(encoding="utf-8"))
        cur = json.loads(current_metrics_path.read_text(encoding="utf-8"))
        keys = sorted(set(prev) & set(cur))
        regressions = []
        for k in keys:
            pv, cv = prev[k], cur[k]
            if isinstance(pv, (int, float)) and isinstance(cv, (int, float)) and pv > cv + 2.0:
                regressions.append((k, pv, cv))
        return {"flag": len(regressions) > 0, "drops": regressions}


def compute_canary_subset(eval_examples_path: Path, out_path: Path, k: int = 50, seed: int = 1337):
    rng = random_from_seed(seed)
    data = json.loads(eval_examples_path.read_text(encoding="utf-8"))
    idx = list(range(len(data)))
    rng.shuffle(idx)
    picks = sorted(idx[: min(k, len(idx))])
    subset = [data[i] for i in picks]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"indices": picks, "seed": seed}
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def random_from_seed(seed: int):  # small helper avoids numpy depend in util
    import random

    r = random.Random(seed)
    return r


def verify_adapter_matches(adapter_dir: Path, expected_sha_yaml: Path) -> bool:
    mi = ModelIntegrity()
    curr = mi.hash_adapter(adapter_dir)
    if not expected_sha_yaml.is_file():
        return False
    payload = yaml.safe_load(expected_sha_yaml.read_text(encoding="utf-8")) or {}
    expected = payload.get("sha256_adapter_weights", "").strip()
    return curr == expected and bool(curr)


def persist_hash(adapter_dir: Path, out_yaml: Path) -> str:
    h = ModelIntegrity().hash_adapter(adapter_dir)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump({"sha256_adapter_weights": h}, allow_unicode=False), encoding="utf-8")
    return h
