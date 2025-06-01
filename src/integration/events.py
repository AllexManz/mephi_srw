"""
Event processing module for SIEM integration.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from elasticsearch import Elasticsearch
from transformers import PreTrainedModel, PreTrainedTokenizer

class EventProcessor:
    """Класс для обработки событий безопасности."""
    
    def __init__(
        self,
        es_client: Elasticsearch,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: Dict[str, Any]
    ):
        self.es_client = es_client
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        
        # Настройки индекса
        self.index_pattern = cfg.elasticsearch.index_pattern
        self.time_field = cfg.elasticsearch.time_field
        self.max_events = cfg.elasticsearch.max_events_per_query
        
        # Настройки модели
        self.max_length = cfg.model.max_length
        self.temperature = cfg.model.temperature
        self.max_new_tokens = cfg.model.max_new_tokens
    
    def get_recent_events(
        self,
        time_window: Optional[timedelta] = None,
        query: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Получение последних событий из Elasticsearch.
        
        Args:
            time_window: Временное окно для поиска событий
            query: Дополнительные параметры поиска
            
        Returns:
            List[Dict[str, Any]]: Список событий
        """
        # Базовый запрос
        search_query = {
            "size": self.max_events,
            "sort": [{self.time_field: "desc"}],
            "query": {
                "bool": {
                    "must": [
                        {"match_all": {}}  # Базовое условие
                    ]
                }
            }
        }
        
        # Добавляем временное окно если указано
        if time_window:
            time_threshold = datetime.utcnow() - time_window
            search_query["query"]["bool"]["must"].append({
                "range": {
                    self.time_field: {
                        "gte": time_threshold.isoformat()
                    }
                }
            })
        
        # Добавляем пользовательский запрос если указан
        if query:
            search_query["query"]["bool"]["must"].extend(query.get("must", []))
            search_query["query"]["bool"]["should"] = query.get("should", [])
            search_query["query"]["bool"]["filter"] = query.get("filter", [])
        
        # Выполняем поиск
        response = self.es_client.search(
            index=self.index_pattern,
            body=search_query
        )
        
        # Извлекаем события
        events = []
        for hit in response["hits"]["hits"]:
            event = hit["_source"]
            event["_id"] = hit["_id"]
            event["_score"] = hit["_score"]
            events.append(event)
        
        return events
    
    def prepare_prompt(self, events: List[Dict[str, Any]]) -> str:
        """
        Подготовка промпта для модели на основе событий.
        
        Args:
            events: Список событий
            
        Returns:
            str: Подготовленный промпт
        """
        # Форматируем события в текст
        events_text = []
        for event in events:
            # Преобразуем событие в строку
            event_str = json.dumps(event, ensure_ascii=False, indent=2)
            events_text.append(event_str)
        
        # Объединяем события
        events_context = "\n\n".join(events_text)
        
        # Формируем промпт
        prompt = f"""### Инструкция: Проанализируй следующие события безопасности и определи, есть ли признаки атаки или подозрительной активности.

### Контекст: {events_context}

### Вход: Определи уровень угрозы (низкий/средний/высокий) и опиши обнаруженные проблемы.

### Ответ:"""
        
        return prompt
    
    def analyze_events(
        self,
        events: List[Dict[str, Any]],
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Анализ событий с помощью модели.
        
        Args:
            events: Список событий для анализа
            device: Устройство для вычислений
            
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        device = device or ("cuda" if self.model.device.type == "cuda" else "cpu")
        
        # Подготавливаем промпт
        prompt = self.prepare_prompt(events)
        
        # Токенизируем вход
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(device)
        
        # Генерируем ответ
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Декодируем ответ
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлекаем ответ модели
        try:
            analysis = generated_text.split("### Ответ:")[-1].strip()
        except IndexError:
            analysis = ""
        
        # Определяем уровень угрозы
        threat_level = "низкий"
        if "высокий" in analysis.lower():
            threat_level = "высокий"
        elif "средний" in analysis.lower():
            threat_level = "средний"
        
        return {
            "analysis": analysis,
            "threat_level": threat_level,
            "events_count": len(events),
            "timestamp": datetime.utcnow().isoformat()
        } 