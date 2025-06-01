"""
Data processing module for preparing training datasets.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from .utils import format_training_text, prepare_example

class SecurityDataProcessor:
    """Класс для обработки и подготовки данных для обучения."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        train_path: Optional[str] = None,
        eval_path: Optional[str] = None,
        test_path: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_path = train_path
        self.eval_path = eval_path
        self.test_path = test_path
    
    def _load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Загрузка данных из JSON файла."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _prepare_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Подготовка примера для обучения."""
        text = format_training_text(
            instruction=example['instruction'],
            input_text=example['input'],
            output=example['output'],
            context=example.get('context')
        )
        
        return {
            "text": text,
            "instruction": example['instruction'],
            "input": example['input'],
            "output": example['output']
        }
    
    def _tokenize_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[int]]:
        """Токенизация текста."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    def prepare_dataset(self) -> DatasetDict:
        """Подготовка датасета для обучения."""
        datasets = {}
        
        # Загружаем и подготавливаем данные
        if self.train_path:
            train_data = [self._prepare_example(ex) for ex in self._load_json(self.train_path)]
            datasets['train'] = Dataset.from_pandas(pd.DataFrame(train_data))
            
        if self.eval_path:
            eval_data = [self._prepare_example(ex) for ex in self._load_json(self.eval_path)]
            datasets['validation'] = Dataset.from_pandas(pd.DataFrame(eval_data))
            
        if self.test_path:
            test_data = [self._prepare_example(ex) for ex in self._load_json(self.test_path)]
            datasets['test'] = Dataset.from_pandas(pd.DataFrame(test_data))
        
        # Создаем DatasetDict
        dataset_dict = DatasetDict(datasets)
        
        # Токенизируем данные
        tokenized_datasets = dataset_dict.map(
            self._tokenize_function,
            batched=True,
            remove_columns=dataset_dict["train"].column_names
        )
        
        return tokenized_datasets
    
    @staticmethod
    def create_example(
        instruction: str,
        input_text: str,
        output: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Создание примера для датасета."""
        return prepare_example(
            instruction=instruction,
            input_text=input_text,
            output=output,
            context=context
        )
    
    def save_examples(self, examples: List[Dict[str, Any]], file_path: str):
        """Сохранение примеров в JSON файл."""
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2) 