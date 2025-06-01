"""
Main SIEM integration module.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import time
from elasticsearch import Elasticsearch
from transformers import PreTrainedModel, PreTrainedTokenizer
from omegaconf import DictConfig

from .events import EventProcessor
from .alerts import AlertHandler

class SIEMIntegration:
    """Основной класс для интеграции с SIEM."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: DictConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        
        # Настройка логирования
        self.logger = self._setup_logger()
        
        # Инициализация клиента Elasticsearch
        self.es_client = self._setup_elasticsearch()
        
        # Инициализация обработчиков
        self.event_processor = EventProcessor(
            es_client=self.es_client,
            model=self.model,
            tokenizer=self.tokenizer,
            cfg=cfg
        )
        self.alert_handler = AlertHandler(
            es_client=self.es_client,
            cfg=cfg,
            log_dir=cfg.logging.log_dir
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера."""
        logger = logging.getLogger("SIEMIntegration")
        logger.setLevel(logging.INFO)
        
        # Форматтер для логов
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Хендлер для вывода в консоль
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Хендлер для записи в файл
        log_path = Path(self.cfg.logging.log_dir) / "siem_integration.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_elasticsearch(self) -> Elasticsearch:
        """Инициализация клиента Elasticsearch."""
        try:
            es_client = Elasticsearch(
                hosts=self.cfg.elasticsearch.hosts,
                basic_auth=(
                    self.cfg.elasticsearch.username,
                    self.cfg.elasticsearch.password
                ),
                verify_certs=self.cfg.elasticsearch.verify_certs
            )
            
            # Проверяем подключение
            if not es_client.ping():
                raise ConnectionError("Failed to connect to Elasticsearch")
            
            self.logger.info("Successfully connected to Elasticsearch")
            return es_client
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            raise
    
    def monitor_events(
        self,
        interval: Optional[int] = None,
        time_window: Optional[int] = None,
        query: Optional[Dict[str, Any]] = None
    ):
        """
        Мониторинг событий безопасности.
        
        Args:
            interval: Интервал проверки в секундах
            time_window: Временное окно в минутах
            query: Дополнительные параметры поиска
        """
        interval = interval or self.cfg.monitoring.interval
        time_window = time_window or self.cfg.monitoring.time_window
        
        self.logger.info(
            f"Starting event monitoring with interval {interval}s "
            f"and time window {time_window}m"
        )
        
        try:
            while True:
                # Получаем последние события
                events = self.event_processor.get_recent_events(
                    time_window=timedelta(minutes=time_window),
                    query=query
                )
                
                if events:
                    self.logger.info(f"Found {len(events)} events to analyze")
                    
                    # Анализируем события
                    analysis = self.event_processor.analyze_events(events)
                    
                    # Создаем алерт если уровень угрозы выше низкого
                    if analysis["threat_level"] != "низкий":
                        alert = self.alert_handler.create_alert(
                            analysis=analysis,
                            events=events,
                            metadata={"query": query}
                        )
                        self.logger.info(
                            f"Created alert {alert['_id']} "
                            f"with threat level: {alert['threat_level']}"
                        )
                    else:
                        self.logger.info("No threats detected")
                
                # Ждем следующей проверки
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Event monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error during event monitoring: {str(e)}")
            raise
    
    def analyze_historical_events(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        query: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Анализ исторических событий.
        
        Args:
            start_time: Время начала периода
            end_time: Время окончания периода
            query: Дополнительные параметры поиска
            
        Returns:
            List[Dict[str, Any]]: Список созданных алертов
        """
        end_time = end_time or datetime.utcnow()
        
        self.logger.info(
            f"Analyzing historical events from {start_time} to {end_time}"
        )
        
        # Формируем запрос с временным диапазоном
        time_query = {
            "range": {
                self.event_processor.time_field: {
                    "gte": start_time.isoformat(),
                    "lte": end_time.isoformat()
                }
            }
        }
        
        if query is None:
            query = {"must": [time_query]}
        else:
            query["must"] = query.get("must", []) + [time_query]
        
        # Получаем события
        events = self.event_processor.get_recent_events(query=query)
        
        if not events:
            self.logger.info("No events found in the specified time range")
            return []
        
        self.logger.info(f"Found {len(events)} events to analyze")
        
        # Анализируем события
        analysis = self.event_processor.analyze_events(events)
        
        # Создаем алерт
        alert = self.alert_handler.create_alert(
            analysis=analysis,
            events=events,
            metadata={
                "query": query,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
        )
        
        self.logger.info(
            f"Created historical alert {alert['_id']} "
            f"with threat level: {alert['threat_level']}"
        )
        
        return [alert]
    
    def get_recent_alerts(
        self,
        threat_level: Optional[str] = None,
        time_window: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Получение последних алертов.
        
        Args:
            threat_level: Уровень угрозы для фильтрации
            time_window: Временное окно в минутах
            limit: Максимальное количество алертов
            
        Returns:
            List[Dict[str, Any]]: Список алертов
        """
        alerts = self.alert_handler.get_alerts(
            threat_level=threat_level,
            time_window=time_window,
            limit=limit
        )
        
        self.logger.info(f"Retrieved {len(alerts)} alerts")
        return alerts
    
    def update_alert(
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
        success = self.alert_handler.update_alert_status(
            alert_id=alert_id,
            status=status,
            comment=comment
        )
        
        if success:
            self.logger.info(f"Successfully updated alert {alert_id}")
        else:
            self.logger.error(f"Failed to update alert {alert_id}")
        
        return success 