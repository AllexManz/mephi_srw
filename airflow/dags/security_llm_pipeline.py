"""Airflow DAG for the security LLM experiment lifecycle.

The tasks call the same CLI used locally, so a run can be reproduced outside
Airflow and each stage is visible as a separate Airflow task.
"""

from datetime import datetime
from pathlib import Path
import shlex
import sys

from airflow import DAG
from airflow.operators.bash import BashOperator


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from runtime_config import load_runtime_config  # noqa: E402


RUNTIME_PATH = REPO_ROOT / "configs" / "runtime.yaml"
RUNTIME = load_runtime_config(RUNTIME_PATH)
CAMPAIGN = RUNTIME["campaign"]
ROOT = str(REPO_ROOT)
MODELS = ",".join(CAMPAIGN["models"])
EXPERIMENT = CAMPAIGN["name"]
PRETRAIN_METHOD = CAMPAIGN["pretrain_methods"][0]
RL_METHOD = CAMPAIGN["rl_methods"][0]
BENCHMARK_LIMIT = str(CAMPAIGN["benchmark"]["limit"])
PYTHON = RUNTIME["project"]["python"]


def run_command(command: str) -> str:
    return f"cd {shlex.quote(ROOT)} && {shlex.quote(PYTHON)} {command}"


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
            f"src/run_experiment.py --config {RUNTIME_PATH} --experiment {EXPERIMENT} --models {MODELS} "
            f"--phase baseline --benchmark-limit {BENCHMARK_LIMIT}"
        ),
    )
    pretrain = BashOperator(
        task_id="pretrain",
        bash_command=run_command(
            f"src/run_experiment.py --config {RUNTIME_PATH} --experiment {EXPERIMENT} --models {MODELS} "
            f"--phase pretrain --pretrain-method {PRETRAIN_METHOD}"
        ),
    )
    rl_alignment = BashOperator(
        task_id="rl_alignment",
        bash_command=run_command(
            f"src/run_experiment.py --config {RUNTIME_PATH} --experiment {EXPERIMENT} --models {MODELS} "
            f"--phase rl --rl-method {RL_METHOD}"
        ),
    )
    post_training_benchmarks = BashOperator(
        task_id="post_training_benchmarks",
        bash_command=run_command(
            f"src/run_experiment.py --config {RUNTIME_PATH} --experiment {EXPERIMENT} --models {MODELS} "
            f"--phase evaluate --benchmark-limit {BENCHMARK_LIMIT}"
        ),
    )

    prepare_data >> baseline >> pretrain >> rl_alignment >> post_training_benchmarks
