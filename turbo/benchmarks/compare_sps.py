#!/usr/bin/env python3
"""Run paired, alternating SPS comparisons between two Python environments."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
from pathlib import Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _percentile(sorted_values: list[float], probability: float) -> float:
    index = round((len(sorted_values) - 1) * probability)
    return sorted_values[index]


def _bootstrap_median_ci(values: list[float]) -> tuple[float, float]:
    rng = random.Random(918)
    medians = []
    for _ in range(10_000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        medians.append(statistics.median(sample))
    medians.sort()
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _run(
    python: Path,
    benchmark: Path,
    *,
    label: str,
    num_envs: str,
    warmup_steps: int,
    measured_steps: int,
) -> dict:
    command = [
        str(python),
        str(benchmark),
        "--label",
        label,
        "--num-envs",
        num_envs,
        "--warmup-steps",
        str(warmup_steps),
        "--measured-steps",
        str(measured_steps),
        "--samples",
        "1",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=benchmark.parent,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--num-envs", default="1,2,4,8,16")
    parser.add_argument("--pairs", type=_positive_int, default=7)
    parser.add_argument("--warmup-steps", type=_positive_int, default=40)
    parser.add_argument("--measured-steps", type=_positive_int, default=250)
    args = parser.parse_args()
    benchmark = Path(__file__).with_name("benchmark_sps.py").resolve()

    paired: dict[str, list[dict[str, float]]] = {}
    provenance = {}
    for pair in range(args.pairs):
        labels = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        results = {}
        for label in labels:
            python = (
                args.baseline_python if label == "baseline" else args.candidate_python
            ).absolute()
            result = _run(
                python,
                benchmark,
                label=label,
                num_envs=args.num_envs,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
            )
            provenance[label] = {
                "python": str(python),
                "package_path": result["package_path"],
                "package_version": result["package_version"],
            }
            results[label] = result
        for num_envs, baseline_run in results["baseline"]["runs"].items():
            baseline = baseline_run["samples"][0]
            candidate = results["candidate"]["runs"][num_envs]["samples"][0]
            paired.setdefault(num_envs, []).append(
                {
                    "baseline_sps": baseline,
                    "candidate_sps": candidate,
                    "ratio": candidate / baseline,
                }
            )

    summary = {}
    passed = True
    for num_envs, pairs in paired.items():
        ratios = [pair["ratio"] for pair in pairs]
        lower, upper = _bootstrap_median_ci(ratios)
        median_baseline = statistics.median(pair["baseline_sps"] for pair in pairs)
        median_candidate = statistics.median(pair["candidate_sps"] for pair in pairs)
        lane_passed = median_candidate > median_baseline
        passed &= lane_passed
        summary[num_envs] = {
            "baseline_median_sps": median_baseline,
            "candidate_median_sps": median_candidate,
            "median_ratio": statistics.median(ratios),
            "bootstrap_95pct_median_ratio": [lower, upper],
            "candidate_faster": lane_passed,
            "pairs": pairs,
        }
    print(
        json.dumps(
            {
                "schema_version": 1,
                "passed": passed,
                "provenance": provenance,
                "settings": {
                    "pairs": args.pairs,
                    "num_envs": args.num_envs,
                    "warmup_steps": args.warmup_steps,
                    "measured_steps": args.measured_steps,
                    "alternating_order": True,
                },
                "results": summary,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
