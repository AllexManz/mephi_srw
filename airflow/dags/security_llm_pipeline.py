"""Airflow DAG for the security LLM experiment lifecycle.

The tasks call the same CLI used locally, so a run can be reproduced outside
Airflow and each stage is visible as a separate Airflow task.
"""

from datetime import datetime
import os

from airflow import DAG
from airflow.operators.bash import BashOperator


ROOT = os.environ.get("SECURITY_LLM_REPO", "/opt/security-llm")
MODELS = os.environ.get("SECURITY_LLM_MODELS", "gpt2")
EXPERIMENT = os.environ.get("SECURITY_LLM_EXPERIMENT", "security-llm")
PRETRAIN_METHOD = os.environ.get("SECURITY_LLM_PRETRAIN", "lora")
RL_METHOD = os.environ.get("SECURITY_LLM_RL", "gspo")
BENCHMARK_LIMIT = os.environ.get("SECURITY_LLM_BENCHMARK_LIMIT", "30")
PYTHON = os.environ.get("SECURITY_LLM_PYTHON", "python")


def run_command(command: str) -> str:
    return f"cd {ROOT} && PYTHONPATH=src {PYTHON} {command}"


with DAG(
    dag_id="security_llm_experiment",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["security-llm", "finetuning", "mlflow"],
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=run_command("src/data/prepare_dataset.py dataset=small"),
    )
    baseline = BashOperator(
        task_id="baseline_benchmarks",
        bash_command=run_command(
            f"src/run_experiment.py --experiment {EXPERIMENT} --models {MODELS} "
            f"--phase baseline --benchmark-limit {BENCHMARK_LIMIT}"
        ),
    )
    pretrain = BashOperator(
        task_id="pretrain",
        bash_command=run_command(
            f"src/run_experiment.py --experiment {EXPERIMENT} --models {MODELS} "
            f"--phase pretrain --pretrain-method {PRETRAIN_METHOD}"
        ),
    )
    rl_alignment = BashOperator(
        task_id="rl_alignment",
        bash_command=run_command(
            f"src/run_experiment.py --experiment {EXPERIMENT} --models {MODELS} "
            f"--phase rl --rl-method {RL_METHOD}"
        ),
    )
    post_training_benchmarks = BashOperator(
        task_id="post_training_benchmarks",
        bash_command=run_command(
            f"src/run_experiment.py --experiment {EXPERIMENT} --models {MODELS} "
            f"--phase evaluate --benchmark-limit {BENCHMARK_LIMIT}"
        ),
    )

    prepare_data >> baseline >> pretrain >> rl_alignment >> post_training_benchmarks
