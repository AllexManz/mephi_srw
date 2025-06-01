"""
Alert handling module for SIEM integration.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from elasticsearch import Elasticsearch

class AlertHandler:
    """Класс для обработки и сохранения алертов."""
    
    def __init__(
        self,
        es_client: Elasticsearch,
        cfg: Dict[str, Any],
        log_dir: Optional[str] = None
    ):
        self.es_client = es_client
        self.cfg = cfg
        
        # Настройки индекса для алертов
        self.alert_index = cfg.alerts.index_name
        self.alert_mapping = cfg.alerts.mapping
        
        # Настройки логирования
        self.logger = self._setup_logger(log_dir)
        
        # Создаем индекс для алертов если его нет
        self._ensure_alert_index()
    
    def _setup_logger(self, log_dir: Optional[str] = None) -> logging.Logger:
        """Настройка логгера."""
        logger = logging.getLogger("AlertHandler")
        logger.setLevel(logging.INFO)
        
        # Форматтер для логов
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Хендлер для вывода в консоль
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Хендлер для записи в файл если указана директория
        if log_dir:
            log_path = Path(log_dir) / "alerts.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _ensure_alert_index(self):
        """Создание индекса для алертов если он не существует."""
        if not self.es_client.indices.exists(index=self.alert_index):
            self.es_client.indices.create(
                index=self.alert_index,
                body=self.alert_mapping
            )
            self.logger.info(f"Created alert index: {self.alert_index}")
    
    def create_alert(
        self,
        analysis: Dict[str, Any],
        events: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Создание алерта на основе анализа событий.
        
        Args:
            analysis: Результаты анализа событий
            events: Список событий, вызвавших алерт
            metadata: Дополнительные метаданные
            
        Returns:
            Dict[str, Any]: Созданный алерт
        """
        # Формируем алерт
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "threat_level": analysis["threat_level"],
            "analysis": analysis["analysis"],
            "events_count": analysis["events_count"],
            "events": events,
            "metadata": metadata or {}
        }
        
        # Сохраняем алерт в Elasticsearch
        response = self.es_client.index(
            index=self.alert_index,
            body=alert
        )
        
        # Добавляем ID алерта
        alert["_id"] = response["_id"]
        
        # Логируем создание алерта
        self.logger.info(
            f"Created alert {alert['_id']} with threat level: {alert['threat_level']}"
        )
        
        return alert
    
    def get_alerts(
        self,
        threat_level: Optional[str] = None,
        time_window: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Получение алертов из Elasticsearch.
        
        Args:
            threat_level: Уровень угрозы для фильтрации
            time_window: Временное окно в минутах
            limit: Максимальное количество алертов
            
        Returns:
            List[Dict[str, Any]]: Список алертов
        """
        # Базовый запрос
        query = {
            "size": limit,
            "sort": [{"timestamp": "desc"}],
            "query": {
                "bool": {
                    "must": [
                        {"match_all": {}}
                    ]
                }
            }
        }
        
        # Добавляем фильтр по уровню угрозы
        if threat_level:
            query["query"]["bool"]["must"].append({
                "term": {"threat_level": threat_level}
            })
        
        # Добавляем временное окно
        if time_window:
            time_threshold = datetime.utcnow().timestamp() - (time_window * 60)
            query["query"]["bool"]["must"].append({
                "range": {
                    "timestamp": {
                        "gte": datetime.fromtimestamp(time_threshold).isoformat()
                    }
                }
            })
        
        # Выполняем поиск
        response = self.es_client.search(
            index=self.alert_index,
            body=query
        )
        
        # Извлекаем алерты
        alerts = []
        for hit in response["hits"]["hits"]:
            alert = hit["_source"]
            alert["_id"] = hit["_id"]
            alert["_score"] = hit["_score"]
            alerts.append(alert)
        
        return alerts
    
    def update_alert_status(
        self,
        alert_id: str,
        status: str,
        comment: Optional[str] = None
    ) -> bool:
        """
        Обновление статуса алерта.
        
        Args:
            alert_id: ID алерта
            status: Новый статус
            comment: Комментарий к обновлению
            
        Returns:
            bool: True если обновление успешно
        """
        try:
            # Обновляем алерт
            self.es_client.update(
                index=self.alert_index,
                id=alert_id,
                body={
                    "doc": {
                        "status": status,
                        "status_updated_at": datetime.utcnow().isoformat(),
                        "status_comment": comment
                    }
                }
            )
            
            self.logger.info(f"Updated alert {alert_id} status to: {status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update alert {alert_id}: {str(e)}")
            return False 