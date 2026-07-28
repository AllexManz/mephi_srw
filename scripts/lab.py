#!/usr/bin/env python3
"""Single entry point for services, experiments, evaluation and reports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from runtime_config import load_runtime_config, resolve_path  # noqa: E402


class LabError(RuntimeError):
    """User-facing launcher error."""


class Lab:
    def __init__(self, config_path: str):
        self.config_path = str(resolve_path(config_path))
        self.config = load_runtime_config(self.config_path)
        self.runtime_dir = resolve_path(self.config["project"]["runtime_dir"])
        self.logs_dir = self.runtime_dir / "services" / "logs"
        bootstrap_python = self.config["project"].get("bootstrap_python", "python3")
        self.bootstrap_python = shutil.which(bootstrap_python) or bootstrap_python
        self.python = str(resolve_path(self.config["project"]["python"]))
        self.mlflow_python = str(resolve_path(self.config["project"]["mlflow_python"]))
        self.airflow_python = str(resolve_path(self.config["project"]["airflow_python"]))

    def init(self) -> None:
        directories = [
            self.runtime_dir,
            self.logs_dir,
            self.runtime_dir / "services" / "pids",
            resolve_path(self.config["services"]["mlflow"]["backend_store"]).parent,
            resolve_path(self.config["services"]["mlflow"]["artifact_root"]),
            resolve_path(self.config["services"]["airflow"]["home"]),
            ROOT / ".secrets",
            ROOT / "reports",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        secret_specs = {
            "minio_root_user": "security-llm-local",
            "minio_root_password": secrets.token_urlsafe(32),
        }
        for name, value in secret_specs.items():
            path = ROOT / ".secrets" / name
            if not path.exists():
                path.write_text(value + "\n", encoding="utf-8")
                path.chmod(0o600)
        print(f"Runtime initialized at {self.runtime_dir}")

    def bootstrap(self) -> None:
        """Create an isolated environment for orchestration and UI services."""
        training_python_path = Path(self.python)
        if not training_python_path.exists():
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    self.bootstrap_python,
                    "-m",
                    "venv",
                    "--system-site-packages",
                    str(training_python_path.parents[1]),
                ],
                cwd=ROOT,
                check=True,
            )
        subprocess.run(
            [
                self.python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--index-url",
                "https://pypi.org/simple",
                "mlflow-skinny==2.19.0",
                "pyarrow>=21,<25",
            ],
            cwd=ROOT,
            check=True,
        )
        # Reinstall only the shared namespace files. This also repairs a venv
        # after replacing the full mlflow package with mlflow-skinny.
        subprocess.run(
            [
                self.python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--index-url",
                "https://pypi.org/simple",
                "--force-reinstall",
                "--no-deps",
                "mlflow-skinny==2.19.0",
            ],
            cwd=ROOT,
            check=True,
        )
        mlflow_python_path = Path(self.mlflow_python)
        if not mlflow_python_path.exists():
            subprocess.run(
                [
                    self.bootstrap_python,
                    "-m",
                    "venv",
                    str(mlflow_python_path.parents[1]),
                ],
                cwd=ROOT,
                check=True,
            )
        subprocess.run(
            [
                self.mlflow_python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--index-url",
                "https://pypi.org/simple",
                "mlflow==2.19.0",
            ],
            cwd=ROOT,
            check=True,
        )
        airflow_python_path = Path(self.airflow_python)
        if not airflow_python_path.exists():
            subprocess.run(
                [
                    self.bootstrap_python,
                    "-m",
                    "venv",
                    str(airflow_python_path.parents[1]),
                ],
                cwd=ROOT,
                check=True,
            )
        subprocess.run(
            [
                self.airflow_python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--index-url",
                "https://pypi.org/simple",
                "apache-airflow==2.10.4",
                "--constraint",
                "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.4/constraints-3.10.txt",
            ],
            cwd=ROOT,
            check=True,
        )
        print(f"Training/MLflow environment ready: {self.python}")
        print(f"MLflow server environment ready: {self.mlflow_python}")
        print(f"Airflow environment ready: {self.airflow_python}")

    def _module_available(self, python: str, module: str) -> bool:
        check = subprocess.run(
            [python, "-c", f"import importlib; importlib.import_module('{module}')"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return check.returncode == 0

    def _airflow_env(self) -> Dict[str, str]:
        """Build Airflow's process-only environment from runtime.yaml."""
        airflow = self.config["services"]["airflow"]
        return {
            **os.environ,
            "AIRFLOW_HOME": str(resolve_path(airflow["home"])),
            "AIRFLOW__CORE__DAGS_FOLDER": str(resolve_path(airflow["dags_folder"])),
            "AIRFLOW__CORE__LOAD_EXAMPLES": str(
                airflow.get("load_examples", False)
            ).lower(),
            "AIRFLOW__WEBSERVER__WEB_SERVER_HOST": str(airflow["host"]),
            "AIRFLOW__WEBSERVER__WEB_SERVER_PORT": str(airflow["port"]),
            "PATH": f"{Path(self.airflow_python).parent}:{os.environ.get('PATH', '')}",
        }

    def doctor(self, require_gpu: bool = False) -> bool:
        checks: List[tuple[str, bool, str]] = []
        training_python_exists = Path(self.python).exists()
        checks.append(("training virtualenv", training_python_exists, self.python))
        for module in ("torch", "transformers", "peft", "datasets", "yaml", "tensorboard", "mlflow"):
            ok = training_python_exists and self._module_available(self.python, module)
            checks.append((f"training module {module}", ok, "installed" if ok else "missing"))
        mlflow_python_exists = Path(self.mlflow_python).exists()
        checks.append(("MLflow server virtualenv", mlflow_python_exists, self.mlflow_python))
        mlflow_server_ok = mlflow_python_exists and self._module_available(self.mlflow_python, "mlflow")
        checks.append(("MLflow server module", mlflow_server_ok, "installed" if mlflow_server_ok else "missing"))
        airflow_python_exists = Path(self.airflow_python).exists()
        checks.append(("Airflow virtualenv", airflow_python_exists, self.airflow_python))
        airflow_ok = False
        if airflow_python_exists:
            airflow_check = subprocess.run(
                [
                    self.airflow_python,
                    "-c",
                    "from importlib.metadata import version; print(version('apache-airflow'))",
                ],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            airflow_ok = airflow_check.returncode == 0
        checks.append(("Airflow module", airflow_ok, "installed" if airflow_ok else "missing"))
        if airflow_ok:
            dag_check = subprocess.run(
                [
                    self.airflow_python,
                    "-m",
                    "airflow",
                    "dags",
                    "list-import-errors",
                    "--output",
                    "json",
                ],
                cwd=ROOT,
                env=self._airflow_env(),
                text=True,
                capture_output=True,
            )
            dag_output = dag_check.stdout.strip()
            dags_ok = dag_check.returncode == 0 and dag_output == "[]"
            checks.append((
                "Airflow DAG imports",
                dags_ok,
                "no errors" if dags_ok else (dag_output or dag_check.stderr.strip()),
            ))

        gpu = subprocess.run(
            [self.python, "-c", "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        gpu_ok = gpu.returncode == 0 and gpu.stdout.strip().startswith("True")
        checks.append(("CUDA", gpu_ok or not require_gpu, gpu.stdout.strip() or gpu.stderr.strip()))

        usage = shutil.disk_usage(ROOT)
        free_gib = usage.free / 1024**3
        checks.append(("free disk >= 50 GiB", free_gib >= 50, f"{free_gib:.1f} GiB"))

        valid = True
        for name, ok, detail in checks:
            marker = "OK" if ok else "FAIL"
            print(f"[{marker}] {name}: {detail}")
            valid = valid and ok
        return valid

    def _pid_path(self, service: str) -> Path:
        return self.runtime_dir / "services" / "pids" / f"{service}.pid"

    def _read_pid(self, service: str) -> Optional[int]:
        path = self._pid_path(service)
        if not path.exists():
            return None
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            path.unlink(missing_ok=True)
            return None

    def _start(
        self,
        service: str,
        command: List[str],
        *,
        internal_env: Optional[Dict[str, str]] = None,
    ) -> None:
        existing = self._read_pid(service)
        if existing:
            print(f"{service}: already running (pid {existing})")
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{service}.log"
        log = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        if internal_env:
            env.update(internal_env)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self._pid_path(service).write_text(str(process.pid), encoding="utf-8")
        print(f"{service}: started (pid {process.pid}, log {log_path})")

    @staticmethod
    def _wait_port(host: str, port: int, timeout: int) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except OSError:
                time.sleep(1)
        return False

    def up(self) -> None:
        self.init()
        mlflow = self.config["services"]["mlflow"]
        if mlflow.get("enabled", True):
            backend = resolve_path(mlflow["backend_store"])
            artifacts = resolve_path(mlflow["artifact_root"])
            if self._wait_port(str(mlflow["host"]), int(mlflow["port"]), 1):
                print("mlflow: healthy external/existing service is already listening")
            else:
                self._start(
                    "mlflow",
                    [
                        self.mlflow_python,
                        "-m",
                        "mlflow",
                        "server",
                        "--host",
                        str(mlflow["host"]),
                        "--port",
                        str(mlflow["port"]),
                        "--backend-store-uri",
                        f"sqlite:///{backend}",
                        "--default-artifact-root",
                        artifacts.as_uri(),
                    ],
                )
            if not self._wait_port(str(mlflow["host"]), int(mlflow["port"]), 45):
                raise LabError(f"MLflow did not open port {mlflow['port']}; see service log")

        tensorboard = self.config["services"]["tensorboard"]
        if tensorboard.get("enabled", True):
            if self._wait_port(
                str(tensorboard["host"]), int(tensorboard["port"]), 1
            ):
                print("tensorboard: healthy external/existing service is already listening")
            else:
                self._start(
                    "tensorboard",
                    [
                        self.python,
                        "-m",
                        "tensorboard.main",
                        "--logdir",
                        str(resolve_path(tensorboard["log_dir"])),
                        "--host",
                        str(tensorboard["host"]),
                        "--port",
                        str(tensorboard["port"]),
                    ],
                )
            if not self._wait_port(str(tensorboard["host"]), int(tensorboard["port"]), 30):
                raise LabError("TensorBoard did not open its configured port; see service log")

        airflow = self.config["services"]["airflow"]
        if airflow.get("enabled", True):
            # Airflow itself only accepts these bootstrapping values through its
            # process environment. They are derived from runtime.yaml internally;
            # users never need to export or persist environment variables.
            if self._wait_port(str(airflow["host"]), int(airflow["port"]), 1):
                print("airflow: healthy external/existing service is already listening")
            else:
                self._start(
                    "airflow",
                    [self.airflow_python, "-m", "airflow", "standalone"],
                    internal_env=self._airflow_env(),
                )
            if not self._wait_port(str(airflow["host"]), int(airflow["port"]), 45):
                raise LabError("Airflow did not open its configured port; see service log")
        print("Services started. Use `python scripts/lab.py status` for details.")

    def down(self) -> None:
        for service in ("airflow", "tensorboard", "mlflow"):
            pid = self._read_pid(service)
            if not pid:
                print(f"{service}: not running")
                continue
            os.killpg(pid, signal.SIGTERM)
            for _ in range(20):
                if not self._read_pid(service):
                    break
                time.sleep(0.25)
            self._pid_path(service).unlink(missing_ok=True)
            print(f"{service}: stopped")

    def status(self) -> None:
        configs = self.config["services"]
        for service in ("mlflow", "tensorboard", "airflow"):
            pid = self._read_pid(service)
            if pid:
                state = f"running (pid {pid})"
            elif self._wait_port(
                str(configs[service]["host"]), int(configs[service]["port"]), 1
            ):
                state = "available (external/existing process)"
            else:
                state = "stopped"
            print(f"{service}: {state}")
        mlflow = self.config["services"]["mlflow"]
        airflow = self.config["services"]["airflow"]
        tensorboard = self.config["services"]["tensorboard"]
        print(f"MLflow UI: http://{mlflow['host']}:{mlflow['port']}")
        print(f"Airflow UI: http://{airflow['host']}:{airflow['port']}")
        print(f"TensorBoard: http://{tensorboard['host']}:{tensorboard['port']}")

    def campaign(self, phase: str, dry_run: bool, best_model: Optional[str]) -> None:
        command = [
            self.python,
            "src/run_campaign.py",
            "--config",
            self.config_path,
            "--phase",
            phase,
        ]
        if dry_run:
            command.append("--dry-run")
        if best_model:
            command.extend(["--best-model", best_model])
        subprocess.run(command, cwd=ROOT, check=True)

    def report(self) -> None:
        subprocess.run(
            [
                self.python,
                "-m",
                "evaluation.generate_academic_report",
                "--config",
                self.config_path,
                "--output",
                "reports/EXPERIMENT_REPORT.md",
            ],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "bootstrap", "doctor", "up", "down", "status", "baseline", "matrix", "report", "all"))
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--best-model")
    parser.add_argument("--skip-services", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    lab = Lab(args.config)

    if args.command == "init":
        lab.init()
    elif args.command == "bootstrap":
        lab.init()
        lab.bootstrap()
    elif args.command == "doctor":
        raise SystemExit(0 if lab.doctor(args.require_gpu) else 1)
    elif args.command == "up":
        lab.up()
    elif args.command == "down":
        lab.down()
    elif args.command == "status":
        lab.status()
    elif args.command in ("baseline", "matrix"):
        lab.campaign(args.command, args.dry_run, args.best_model)
        if not args.dry_run:
            lab.report()
    elif args.command == "report":
        lab.report()
    elif args.command == "all":
        lab.init()
        if not args.dry_run and (
            not Path(lab.python).exists()
            or not Path(lab.mlflow_python).exists()
            or not Path(lab.airflow_python).exists()
        ):
            lab.bootstrap()
        if not args.dry_run and not lab.doctor(require_gpu=True):
            raise LabError("Preflight failed; fix missing dependencies/GPU before a real campaign")
        if not args.skip_services and not args.dry_run:
            lab.up()
        lab.campaign("all", args.dry_run, args.best_model)
        if not args.dry_run:
            lab.report()


if __name__ == "__main__":
    main()
