#!/usr/bin/env python3
"""Convenience wrapper that chains mock-friendly smoke tests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Smoke-test generators + mock benchmarks")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    cmds = [
        [sys.executable, str(REPO / "security_testing/datasets/generate_prompt_injection.py")],
        [sys.executable, str(REPO / "security_testing/datasets/generate_data_leakage.py")],
        [sys.executable, str(REPO / "security_testing/datasets/generate_poisoned_data.py")],
        [
            sys.executable,
            str(REPO / "security_testing/attacks/run_prompt_injection.py"),
            "--mock-model",
            f"--limit={args.limit}",
            "--output",
            str(REPO / "security_testing/results/pi_baseline_mock.json"),
        ],
        [
            sys.executable,
            str(REPO / "security_testing/attacks/run_prompt_injection.py"),
            "--mock-model",
            "--with-defense",
            f"--limit={args.limit}",
            "--output",
            str(REPO / "security_testing/results/pi_defense_mock.json"),
        ],
        [
            sys.executable,
            str(REPO / "security_testing/attacks/run_data_leakage.py"),
            "--mock-model",
            f"--limit={args.limit}",
            "--output",
            str(REPO / "security_testing/results/dl_baseline_mock.json"),
        ],
        [
            sys.executable,
            str(REPO / "security_testing/attacks/run_data_leakage.py"),
            "--mock-model",
            "--with-defense",
            f"--limit={args.limit}",
            "--output",
            str(REPO / "security_testing/results/dl_defense_mock.json"),
        ],
        [
            sys.executable,
            str(REPO / "security_testing/attacks/run_poisoning.py"),
            "--mock-model",
            "--output",
            str(REPO / "security_testing/results/poison_screen_mock.json"),
        ],
        [
            sys.executable,
            str(REPO / "security_testing/evaluation/compare_results.py"),
            "--pi-base",
            str(REPO / "security_testing/results/pi_baseline_mock.json"),
            "--pi-sec",
            str(REPO / "security_testing/results/pi_defense_mock.json"),
            "--dl-base",
            str(REPO / "security_testing/results/dl_baseline_mock.json"),
            "--dl-sec",
            str(REPO / "security_testing/results/dl_defense_mock.json"),
            "--poison-base",
            str(REPO / "security_testing/results/poison_screen_mock.json"),
            "--poison-def",
            str(REPO / "security_testing/results/poison_screen_mock.json"),
            "--out",
            str(REPO / "security_testing/results/final_comparison_mock.json"),
        ],
    ]

    for cmd in cmds:
        print("running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(REPO), check=False)

    out = REPO / "security_testing/results/final_comparison_mock.json"
    print("done; summary at", out)
    if out.is_file():
        print(json.loads(out.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
