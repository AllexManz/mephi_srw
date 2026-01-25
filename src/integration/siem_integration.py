"""
Standalone SIEM integration helper.
"""

from typing import List, Dict, Any, Optional, Callable
import json
from datetime import datetime, timedelta
import time
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from omegaconf import DictConfig


class SIEMIntegration:
    """Класс для простой интеграции с SIEM через Elasticsearch."""

    def __init__(
        self,
        model_path: str,
        cfg: DictConfig,
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
            hosts=cfg.elasticsearch.hosts,
            basic_auth=(
                cfg.elasticsearch.username,
                cfg.elasticsearch.password
            ) if cfg.elasticsearch.username and cfg.elasticsearch.password else None,
            verify_certs=cfg.elasticsearch.verify_certs
        )

        self.index_pattern = cfg.elasticsearch.index_pattern
        self.max_events = cfg.elasticsearch.max_events_per_query
        self.time_window = cfg.monitoring.time_window

        # Настройки генерации
        eval_cfg = cfg.evaluation.evaluation if "evaluation" in cfg.evaluation else cfg.evaluation
        gen_cfg = eval_cfg.generation if "generation" in eval_cfg else eval_cfg
        self.max_new_tokens = gen_cfg.max_new_tokens
        self.temperature = eval_cfg.temperature

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
            f"Event {i + 1}:\n" + json.dumps(event, ensure_ascii=False, indent=2)
            for i, event in enumerate(events)
        ])

        return (
            "### Инструкция: Проанализируй следующие события безопасности и "
            "предоставь краткий анализ и рекомендации.\n\n"
            f"### События:\n{events_text}\n\n"
            "### Анализ и рекомендации:"
        )

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
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            temperature=self.temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлечение анализа из ответа модели
        if "### Анализ и рекомендации:" in response:
            analysis = response.split("### Анализ и рекомендации:", 1)[1].strip()
        else:
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

    def monitor_security_events(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
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


__all__ = ["SIEMIntegration"]