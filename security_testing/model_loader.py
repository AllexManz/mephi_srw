"""Load causal LM (+ optional QLoRA) matching `src/evaluate.py` patterns."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from security_testing.common.repo_paths import find_repo_root


def setup_sys_path_for_src() -> None:
    repo = find_repo_root()
    sd = repo / "src"
    sd_s = str(sd)
    if sd_s not in sys.path:
        sys.path.insert(0, sd_s)


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)




def load_model_and_tokenizer_from_security_config(device_map: str | None = "auto"):
    """
    Load transformers model + tokenizer.

    Reads security_testing/configs/model_config.yaml for base path and qlora adapter.
    """
    setup_sys_path_for_src()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    repo = find_repo_root()
    mc_path = repo / "security_testing" / "configs" / "model_config.yaml"
    cfg = load_yaml_config(mc_path)
    model_name = cfg["base_model"]

    tokenizer_source = cfg.get("tokenizer_dir")
    tokenizer_path = (repo / tokenizer_source) if tokenizer_source else model_name

    qlora_enabled = cfg.get("qlora", {}).get("enabled", False)
    adapter_path_opt = cfg.get("qlora", {}).get("adapter_path")

    adapters_dir_cfg = cfg.get("adapters_dir_relative")
    default_adapter_dir = repo / adapters_dir_cfg / f"{cfg.get('adapter_name', 'qlora')}_adapter"
    adapter_path = adapter_path_opt or str(default_adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    torch_dtype_str = cfg.get("torch_dtype", "float16")
    torch_dtype = getattr(torch, torch_dtype_str)
    # Loading arbitrary Python from a model repository is equivalent to
    # executing third-party code. Opt in explicitly in model_config.yaml.
    trust_remote_code = cfg.get("trust_remote_code", False)

    quantization_config = None
    qconf = cfg.get("qlora", {}).get("quantization", {})
    if qlora_enabled and qconf.get("load_in_4bit"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, qconf.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_quant_type=qconf.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(qconf.get("bnb_4bit_use_double_quant", True)),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config,
    )

    if qlora_enabled and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer, cfg


