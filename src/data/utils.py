"""
Utility functions for data processing.
"""

from typing import Dict, Any, Optional

def format_training_text(
    instruction: str,
    input_text: str,
    output: str,
    context: Optional[str] = None
) -> str:
    """
    Форматирование текста для обучения в формате инструкций.
    
    Args:
        instruction: Инструкция для модели
        input_text: Входной текст
        output: Ожидаемый выход
        context: Опциональный контекст
        
    Returns:
        str: Отформатированный текст
    """
    return f"""### Инструкция: {instruction}

### Контекст: {context or ''}

### Вход: {input_text}

### Ответ: {output}"""

def prepare_example(
    instruction: str,
    input_text: str,
    output: str,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Подготовка примера для датасета.
    
    Args:
        instruction: Инструкция для модели
        input_text: Входной текст
        output: Ожидаемый выход
        context: Опциональный контекст
        
    Returns:
        Dict[str, Any]: Подготовленный пример
    """
    example = {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }
    if context:
        example["context"] = context
    return example 