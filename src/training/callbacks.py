"""
Callbacks for model training.
"""

import math
from transformers import TrainerCallback

class PerplexityCallback(TrainerCallback):
    """Callback для вычисления перплексии."""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            perplexity = math.exp(metrics["eval_loss"])
            metrics["eval_perplexity"] = perplexity 