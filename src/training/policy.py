"""
Policy optimization helpers for GRPO/GSPO-style training.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from transformers import Trainer, AutoModelForCausalLM


def create_reference_model(model_name: str, cfg: Any) -> AutoModelForCausalLM:
    """Create a frozen reference model for KL regularization."""
    fsdp_enabled = cfg.training.training.get("fsdp", {}).get("enabled", False)
    device_map = None if fsdp_enabled else cfg.model.model.device_map
    model_source = cfg.model.model.get("pretrained_path") or model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=getattr(torch, cfg.model.model.torch_dtype),
        device_map=device_map,
        trust_remote_code=cfg.model.model.trust_remote_code
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class PolicyOptimTrainer(Trainer):
    """Trainer with GRPO/GSPO-style KL regularization."""

    def __init__(
        self,
        *args,
        ref_model: Optional[AutoModelForCausalLM] = None,
        policy_method: str = "grpo",
        policy_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.policy_method = policy_method
        self.policy_config = policy_config or {}
        if self.ref_model is not None:
            self.ref_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.ref_model is None:
            return (loss, outputs) if return_outputs else loss

        # Ensure reference model is on the same device
        ref_device = next(model.parameters()).device
        if next(self.ref_model.parameters()).device != ref_device:
            self.ref_model.to(ref_device)

        logits = outputs.logits
        ref_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            ref_logits = self.ref_model(**ref_inputs).logits

        # Align logits with next-token prediction
        logits = logits[:, :-1, :]
        ref_logits = ref_logits[:, :-1, :]

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            mask = torch.ones(logits.shape[:2], device=logits.device)
        else:
            mask = attention_mask[:, 1:].to(logits.device)

        temperature = float(self.policy_config.get("temperature", 1.0))
        kl_coef = float(self.policy_config.get("kl_coef", 0.1))

        log_p = F.log_softmax(logits / temperature, dim=-1)
        log_q = F.log_softmax(ref_logits / temperature, dim=-1)

        kl_pq = F.kl_div(log_p, log_q, log_target=True, reduction="none").sum(-1)

        if self.policy_method == "gspo":
            kl_qp = F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(-1)
            kl_term = 0.5 * (kl_pq + kl_qp)
        else:
            kl_term = kl_pq

        kl_term = (kl_term * mask).sum() / mask.sum().clamp_min(1.0)

        total_loss = loss + kl_coef * kl_term
        return (total_loss, outputs) if return_outputs else total_loss
