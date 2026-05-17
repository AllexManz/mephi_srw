#!/usr/bin/env python3
"""Aggregate JSON reports written by attack runners."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def read_bundle(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-base", type=Path)
    parser.add_argument("--pi-sec", type=Path)
    parser.add_argument("--dl-base", type=Path)
    parser.add_argument("--dl-sec", type=Path)
    parser.add_argument("--poison-base", type=Path)
    parser.add_argument("--poison-def", type=Path)
    parser.add_argument("--out", type=Path, default=REPO / "security_testing/results/final_comparison.json")
    args = parser.parse_args()

    bundle = {}
    if args.pi_base and args.pi_base.is_file():
        bundle["prompt_injection_baseline_metrics"] = read_bundle(args.pi_base).get("metrics")

    if args.pi_sec and args.pi_sec.is_file():
        bundle["prompt_injection_protected_metrics"] = read_bundle(args.pi_sec).get("metrics")

    if args.dl_base and args.dl_base.is_file():
        bundle["data_leakage_baseline_summary"] = read_bundle(args.dl_base).get("summary")

    if args.dl_sec and args.dl_sec.is_file():
        bundle["data_leakage_protected_summary"] = read_bundle(args.dl_sec).get("summary")

    if args.poison_base and args.poison_base.is_file():
        bundle["poison_baseline"] = read_bundle(args.poison_base)

    if args.poison_def and args.poison_def.is_file():
        bundle["poison_screened"] = read_bundle(args.poison_def)


    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
