#!/usr/bin/env python3
"""
Script to compare evaluation metrics across different base models.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

def main():
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("Logs directory does not exist. Run some training/evaluation first!")
        return

    # Find all model subdirectories in logs/
    model_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and d.name != "tensorboard" and d.name != "training"]
    
    results = {}
    
    # Рекурсивно ищем все файлы evaluation_metrics.json в директории logs/
    for metrics_file in logs_dir.glob("**/evaluation/evaluation_metrics.json"):
        parts = metrics_file.relative_to(logs_dir).parts
        if len(parts) >= 3:
            model_id = parts[0]
            run_id = parts[1]
            display_name = run_id  # Имя конкретного запуска
        else:
            display_name = parts[0]  # Старый формат
            
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            results[display_name] = metrics
        except Exception as e:
            print(f"Error reading metrics from {metrics_file}: {e}")

    if not results:
        print("No evaluation metrics found in logs/ directory. Make sure to run evaluate.py for your models.")
        return

    print(f"Found evaluation metrics for {len(results)} models: {', '.join(results.keys())}")

    # Define key metrics to display in the comparison table
    key_metrics = [
        ("perplexity", "Perplexity", "{:.4f}"),
        ("accuracy", "Accuracy", "{:.4%}"),
        ("threat_detection_accuracy", "Threat Detection Acc", "{:.4%}"),
        ("false_positive_rate", "False Positive Rate", "{:.4%}"),
        ("false_negative_rate", "False Negative Rate", "{:.4%}"),
        ("expert_preference_rate", "Expert Preference Rate", "{:.4%}"),
        ("mitre_attack_coverage", "MITRE Coverage", "{:.4%}"),
        ("average_response_time", "Avg Response Time (s)", "{:.2f}"),
    ]

    # Build Markdown table
    headers = ["Model ID"] + [label for _, label, _ in key_metrics]
    separator = ["---"] * len(headers)
    
    rows = []
    for model_id, metrics in sorted(results.items()):
        row = [model_id]
        for metric_key, _, fmt in key_metrics:
            val = metrics.get(metric_key)
            if val is not None:
                try:
                    row.append(fmt.format(val))
                except Exception:
                    row.append(str(val))
            else:
                row.append("N/A")
        rows.append(row)

    # Generate Markdown text
    markdown_lines = []
    markdown_text = "## Baseline Model Comparison (Stage 1 PEFT)\n\n"
    markdown_text += "| " + " | ".join(headers) + " |\n"
    markdown_text += "| " + " | ".join(separator) + " |\n"
    for row in rows:
        markdown_text += "| " + " | ".join(row) + " |\n"
    
    print("\n" + markdown_text)

    # Save to files
    comparison_md_path = logs_dir / "baseline_comparison.md"
    comparison_json_path = logs_dir / "baseline_comparison.json"
    
    with open(comparison_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Comparison saved to:\n  - Markdown: {comparison_md_path}\n  - JSON: {comparison_json_path}")

if __name__ == "__main__":
    main()
