"""
Main script for model evaluation.
"""

import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.trainer import SecurityModelTrainer
from evaluation.evaluator import SecurityModelEvaluator
from training.dataset import SecurityDataset

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    """Main function to evaluate the model."""
    print("Loading configuration...")
    print(OmegaConf.to_yaml(cfg))
    
    # Create evaluator instance
    evaluator = SecurityModelEvaluator(cfg)
    
    # Run evaluation
    evaluator.evaluate()
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 