#!/usr/bin/env python3
"""Run paired, alternating SPS comparisons between two Python environments."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
from pathlib import Path

_SCALING_PROFILE = "scaling"
_RLAB_PROFILE = "rlab-32x32"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _environment_assignment(value: str) -> tuple[str, str]:
    name, separator, setting = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("environment values must use NAME=VALUE")
    return name, setting


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
    profile: str,
    num_envs: str,
    warmup_steps: int,
    measured_steps: int,
    environment: dict[str, str],
) -> dict:
    command = [
        str(python),
        str(benchmark),
        "--label",
        label,
        "--profile",
        profile,
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
        env={**os.environ, **environment},
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=(_SCALING_PROFILE, _RLAB_PROFILE),
        default=_SCALING_PROFILE,
    )
    parser.add_argument("--num-envs", default="1,2,4,8,16")
    parser.add_argument("--pairs", type=_positive_int, default=7)
    parser.add_argument("--warmup-steps", type=_positive_int, default=40)
    parser.add_argument("--measured-steps", type=_positive_int, default=250)
    parser.add_argument(
        "--baseline-env",
        action="append",
        default=[],
        type=_environment_assignment,
        metavar="NAME=VALUE",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        type=_environment_assignment,
        metavar="NAME=VALUE",
    )
    args = parser.parse_args()
    benchmark = Path(__file__).with_name("benchmark_sps.py").resolve()
    environments = {
        "baseline": dict(args.baseline_env),
        "candidate": dict(args.candidate_env),
    }

    paired: dict[str, list[dict[str, float]]] = {}
    provenance = {}
    for pair in range(args.pairs):
        labels = (
            ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        )
        results = {}
        for label in labels:
            python = (
                args.baseline_python if label == "baseline" else args.candidate_python
            ).absolute()
            result = _run(
                python,
                benchmark,
                label=label,
                profile=args.profile,
                num_envs=args.num_envs,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                environment=environments[label],
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
        baseline_vector_step_ms = 1000.0 * int(num_envs) / median_baseline
        candidate_vector_step_ms = 1000.0 * int(num_envs) / median_candidate
        lane_passed = median_candidate > median_baseline
        passed &= lane_passed
        summary[num_envs] = {
            "baseline_median_sps": median_baseline,
            "candidate_median_sps": median_candidate,
            "baseline_median_vector_step_ms": baseline_vector_step_ms,
            "candidate_median_vector_step_ms": candidate_vector_step_ms,
            "median_sps_ratio": median_candidate / median_baseline,
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
                    "profile": args.profile,
                    "num_envs": args.num_envs,
                    "warmup_steps": args.warmup_steps,
                    "measured_steps": args.measured_steps,
                    "alternating_order": True,
                    "baseline_env": environments["baseline"],
                    "candidate_env": environments["candidate"],
                },
                "results": summary,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
