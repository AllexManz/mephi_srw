#!/usr/bin/env python3
"""Emit markdown summaries for lab notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return lines


def prompt_injection_table(baseline_path: Path, protected_path: Path) -> List[str]:
    base = read_json(baseline_path).get("metrics", {})
    prot = read_json(protected_path).get("metrics", {})

    hdr = ["attack_type", "cases", "baseline_asr", "protected_asr"]
    body_rows: List[List[str]] = []
    for key in sorted(k for k in base.keys() if k != "overall"):
        bb = base.get(key, {})
        bp = prot.get(key, {})
        body_rows.append(
            [
                key,
                str(bb.get("cases", "")),
                f"{float(bb.get('asr', 0))*100:.1f}%",
                f"{float(bp.get('asr', 0))*100:.1f}%",
            ]
        )
    bo = base.get("overall", {})
    po = prot.get("overall", {})
    body_rows.append(
        [
            "overall",
            str(bo.get("cases", "")),
            f'{float(bo.get("asr", 0))*100:.1f}%',
            f'{float(po.get("asr", 0))*100:.1f}%',
        ]
    )

    lines = ["## Prompt injection attack success rate"]
    lines.extend(markdown_table(hdr, body_rows))
    return lines


def overhead_table(baseline_path: Path, protected_path: Path) -> List[str]:
    bef = read_json(baseline_path).get("timings_summary", {}).get("overall", {})
    aft = read_json(protected_path).get("timings_summary", {}).get("overall", {})
    hdr = ["metric", "baseline", "protected", "delta"]
    rows = [
        [
            "mean_latency_s",
            f"{bef.get('mean_latency_s', 0):.4f}",
            f"{aft.get('mean_latency_s', 0):.4f}",
            f"{aft.get('mean_latency_s', 0) - bef.get('mean_latency_s', 0):.4f}",
        ],
        [
            "p95_latency_s",
            f"{bef.get('p95_latency_s', 0):.4f}",
            f"{aft.get('p95_latency_s', 0):.4f}",
            f"{aft.get('p95_latency_s', 0) - bef.get('p95_latency_s', 0):.4f}",
        ],
    ]
    return ["## Latency envelope (evaluation harness)", *markdown_table(hdr, rows)]


def poisoning_section(screened_report: Path) -> List[str]:
    data = read_json(screened_report)
    tv = data.get("training_validator", {})
    lines = ["## Poisoning screening"]
    lines.append(f"- Rows before validator: `{data.get('count_raw')}` ")
    lines.append(f"- Rows after removals: `{data.get('count_after_screening')}` ")
    cand = tv.get("isolation_candidate_indices", [])
    lines.append(f"- Isolation-flagged IDs (sample depth {len(cand)})")
    bm = data.get("benchmark") or {}
    if bm:
        lines.append("- Benchmark median overlap `{:.3f}`".format(bm.get("median_subtoken_overlap_vs_gold", 0.0)))

    lines.append("")
    lines.append("| χ² stat | χ² p-value |")
    lines.append("| --- | --- |")
    chi = tv.get("chi_square") or {}
    lines.append("| {:.2f} | {} |".format(float(chi.get("statistic", 0)), chi.get("pvalue")))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-base", type=Path)
    parser.add_argument("--pi-sec", type=Path)
    parser.add_argument("--poison", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    parts: List[str] = []

    if args.pi_base and args.pi_sec:
        parts.extend(prompt_injection_table(args.pi_base, args.pi_sec))
        parts.append("")
        parts.extend(overhead_table(args.pi_base, args.pi_sec))

    if args.poison:
        parts.append("")
        parts.extend(poisoning_section(args.poison))

    text = "\n".join(parts)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
