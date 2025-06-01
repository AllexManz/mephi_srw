from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from configs.config import Config, config
import time

class SIEMIntegration:
    def __init__(
        self,
        model_path: str,
        es_host: str = config.siem.es_host,
        es_port: int = config.siem.es_port,
        es_username: Optional[str] = config.siem.es_username,
        es_password: Optional[str] = config.siem.es_password,
        use_lora: bool = True,
        use_8bit: bool = True
    ):
        # Инициализация модели
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if use_lora:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=use_8bit,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.model.load_adapter(f"{model_path}/lora_adapter")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                f"{model_path}/full_model",
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Инициализация Elasticsearch
        self.es = Elasticsearch(
            [f"http://{es_host}:{es_port}"],
            basic_auth=(es_username, es_password) if es_username and es_password else None
        )
        
        self.index_pattern = config.siem.es_index_pattern
        self.max_events = config.siem.max_events_per_query
        self.time_window = config.siem.time_window_minutes
        self.alert_threshold = config.siem.alert_threshold
    
    def _get_recent_events(self) -> List[Dict[str, Any]]:
        """Получение последних событий из SIEM."""
        time_threshold = datetime.utcnow() - timedelta(minutes=self.time_window)
        
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": time_threshold.isoformat()}}}
                    ]
                }
            },
            "size": self.max_events,
            "sort": [{"@timestamp": "desc"}]
        }
        
        response = self.es.search(
            index=self.index_pattern,
            body=query
        )
        
        return [hit["_source"] for hit in response["hits"]["hits"]]
    
    def _prepare_prompt(self, events: List[Dict[str, Any]]) -> str:
        """Подготовка промпта для модели на основе событий."""
        events_text = "\n".join([
            f"Event {i+1}:\n" + json.dumps(event, indent=2)
            for i, event in enumerate(events)
        ])
        
        return f"""### Инструкция: Проанализируй следующие события безопасности и предоставь краткий анализ и рекомендации.

### События:
{events_text}

### Анализ и рекомендации:"""
    
    def analyze_events(self) -> Dict[str, Any]:
        """Анализ последних событий безопасности."""
        events = self._get_recent_events()
        if not events:
            return {
                "status": "no_events",
                "message": "No recent events found",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        prompt = self._prepare_prompt(events)
        
        # Генерация ответа модели
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=2048,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлечение анализа из ответа модели
        try:
            analysis = response.split("### Анализ и рекомендации:")[1].strip()
        except IndexError:
            analysis = response
        
        return {
            "status": "success",
            "events_count": len(events),
            "analysis": analysis,
            "timestamp": datetime.utcnow().isoformat(),
            "events": events[:5]  # Возвращаем только первые 5 событий для краткости
        }
    
    def generate_alert(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Генерация алерта на основе анализа."""
        # Здесь можно добавить логику определения критичности анализа
        # и генерации алерта на основе определенных критериев
        
        # Пример простой логики: если в анализе есть определенные ключевые слова,
        # генерируем алерт
        critical_keywords = ["critical", "urgent", "immediate", "severe", "exploit"]
        
        if any(keyword in analysis["analysis"].lower() for keyword in critical_keywords):
            return {
                "alert_type": "security_analysis",
                "severity": "high",
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": analysis["analysis"],
                "events_count": analysis["events_count"]
            }
        
        return None
    
    def monitor_security_events(self, callback: Optional[callable] = None):
        """Мониторинг событий безопасности в реальном времени."""
        while True:
            try:
                analysis = self.analyze_events()
                if analysis["status"] == "success":
                    alert = self.generate_alert(analysis)
                    if alert and callback:
                        callback(alert)
            except Exception as e:
                print(f"Error in monitoring: {str(e)}")
            
            # Пауза между проверками
            time.sleep(60)  # Проверка каждую минуту 