"""
Main script for running SIEM integration.
"""

import argparse
from datetime import datetime, timedelta
import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.trainer import SecurityModelTrainer
from integration.siem import SIEMIntegration

@hydra.main(version_base=None, config_path="../configs", config_name="integration/default")
def main(cfg: DictConfig) -> None:
    """Основная функция для запуска SIEM интеграции."""
    parser = argparse.ArgumentParser(description="Run SIEM integration")
    parser.add_argument(
        "--mode",
        choices=["monitor", "analyze"],
        default="monitor",
        help="Mode of operation: monitor (continuous) or analyze (historical)"
    )
    parser.add_argument(
        "--start-time",
        type=str,
        help="Start time for historical analysis (ISO format)"
    )
    parser.add_argument(
        "--end-time",
        type=str,
        help="End time for historical analysis (ISO format)"
    )
    parser.add_argument(
        "--time-window",
        type=int,
        help="Time window in minutes for monitoring"
    )
    parser.add_argument(
        "--interval",
        type=int,
        help="Check interval in seconds for monitoring"
    )
    args = parser.parse_args()
    
    # Загружаем модель и токенизатор
    print(f"Loading model from {cfg.model.model.name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model.name,
        torch_dtype=getattr(torch, cfg.model.model.torch_dtype),
        device_map=cfg.model.model.device_map,
        trust_remote_code=cfg.model.model.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model.name)
    
    # Если используется PEFT, загружаем адаптер
    if cfg.peft.enabled:
        print(f"Loading PEFT adapter from {cfg.peft.adapter_path}...")
        trainer = SecurityModelTrainer.load_model(cfg.peft.adapter_path, cfg)
        model = trainer.model
        tokenizer = trainer.tokenizer
    
    # Инициализируем SIEM интеграцию
    siem = SIEMIntegration(model, tokenizer, cfg)
    
    if args.mode == "monitor":
        # Запускаем мониторинг
        print("Starting event monitoring...")
        siem.monitor_events(
            interval=args.interval,
            time_window=args.time_window
        )
    else:
        # Анализируем исторические события
        if not args.start_time:
            raise ValueError("--start-time is required for historical analysis")
        
        start_time = datetime.fromisoformat(args.start_time)
        end_time = (
            datetime.fromisoformat(args.end_time)
            if args.end_time
            else datetime.utcnow()
        )
        
        print(
            f"Analyzing historical events from {start_time} to {end_time}..."
        )
        alerts = siem.analyze_historical_events(
            start_time=start_time,
            end_time=end_time
        )
        
        print(f"\nCreated {len(alerts)} alerts:")
        for alert in alerts:
            print(
                f"Alert {alert['_id']}: "
                f"threat_level={alert['threat_level']}, "
                f"events_count={alert['events_count']}"
            )

if __name__ == "__main__":
    main() 