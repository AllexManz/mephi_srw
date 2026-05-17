#!/usr/bin/env python3
"""Exercise prompt injection cases with optional defenses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from security_testing.attacks.evaluate_attack import aggregate_attack_rates, prompt_injection_succeeded
from security_testing.defenses.defense_pipeline import DefensePipeline
from security_testing.inference_utils import MockResponder, generate_answer, summarize_latencies
from security_testing.prompt_conversion import evaluator_style_prompt, splunk_payload_to_instruction_format


def ensure_src():
    sd = REPO_ROOT / "src"
    if str(sd) not in sys.path:
        sys.path.insert(0, str(sd))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "security_testing/test_cases/pi_dataset.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "security_testing/results/pi_baseline.json",
    )
    parser.add_argument("--mock-model", action="store_true")
    parser.add_argument("--with-defense", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases: List[Dict[str, Any]] = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    pipeline = DefensePipeline() if args.with_defense else None
    mock = MockResponder()

    timing_buckets: Dict[str, List[float]] = {}
    rows = []

    if args.mock_model:
        for case in cases:
            blob = case["splunk_event"]
            sanitized_meta = {}
            if pipeline:
                prompt, sanitized_meta = pipeline.prepare_prompt_from_attack_case(blob)
                sys_prefix = pipeline.system_prefix
            else:
                tmpl = splunk_payload_to_instruction_format(blob, wrap_user_data=False)
                prompt = evaluator_style_prompt(tmpl)
                sys_prefix = ""
            ans_plain = mock((sys_prefix or "") + prompt)
            post_md = {}
            if pipeline:
                ans_plain, post_md = pipeline.postprocess_generation(ans_plain)
            succ = prompt_injection_succeeded(case, ans_plain)
            rows.append(
                {
                    "id": case["id"],
                    "type": case["type"],
                    "attack_success": succ,
                    "model_answer_preview": ans_plain[:1600],
                    "post_defense_meta": post_md,
                    "input_filter_meta": sanitized_meta if sanitized_meta else None,
                    "timings": {"response_time_seconds": 0.0},
                }
            )
            timing_buckets.setdefault(case["type"], []).append(0.0)

    else:
        ensure_src()
        from security_testing.model_loader import load_model_and_tokenizer_from_security_config

        model, tokenizer, mcfg = load_model_and_tokenizer_from_security_config()
        inf = mcfg.get("inference", {})
        ml = int(inf.get("max_length", 512))

        for case in cases:
            blob = case["splunk_event"]
            sanitized_meta = {}
            if pipeline:
                prompt, sanitized_meta = pipeline.prepare_prompt_from_attack_case(blob)
                sys_prefix = pipeline.system_prefix
            else:
                tmpl = splunk_payload_to_instruction_format(blob, wrap_user_data=False)
                prompt = evaluator_style_prompt(tmpl)
                sys_prefix = ""

            ans, timings = generate_answer(
                model,
                tokenizer,
                prompt,
                max_length=ml,
                max_new_tokens=int(inf.get("max_new_tokens", 256)),
                temperature=float(inf.get("temperature", 0.7)),
                system_prefix=sys_prefix if args.with_defense else "",
            )
            post_md = {}
            if pipeline:
                ans, post_md = pipeline.postprocess_generation(ans)
            succ = prompt_injection_succeeded(case, ans_plain)
            rows.append(
                {
                    "id": case["id"],
                    "type": case["type"],
                    "attack_success": succ,
                    "model_answer_preview": ans_plain[:1600],
                    "post_defense_meta": post_md,
                    "input_filter_meta": sanitized_meta if sanitized_meta else None,
                    "timings": timings,
                }
            )
            timing_buckets.setdefault(case["type"], []).append(timings.get("response_time_seconds", 0.0))

    aggregates = aggregate_attack_rates(rows)
    timings_summary = {}
    for t_name, lst in timing_buckets.items():
        timings_summary[t_name] = summarize_latencies(lst)
    timings_summary["overall"] = summarize_latencies([item for lst in timing_buckets.values() for item in lst])

    bundle = {"rows": rows, "metrics": aggregates, "timings_summary": timings_summary, "with_defense": args.with_defense}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(json.dumps(aggregates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
