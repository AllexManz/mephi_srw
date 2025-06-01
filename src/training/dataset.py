"""
Dataset module for security language model training.
"""

from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from data.utils import format_training_text

class SecurityDataset(Dataset):
    """Датасет для обучения модели на примерах безопасности."""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Формируем текст для обучения
        text = format_training_text(
            instruction=example['instruction'],
            input_text=example['input'],
            output=example['output'],
            context=example.get('context')
        )
        
        # Токенизируем текст
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        } 