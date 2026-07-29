#!/usr/bin/env python3
"""Compare deterministic environment traces from two Python environments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _environment_assignment(value: str) -> tuple[str, str]:
    name, separator, setting = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("environment values must use NAME=VALUE")
    return name, setting


def _run(python: Path, trace_script: Path, environment: dict[str, str]) -> dict:
    completed = subprocess.run(
        [str(python.absolute()), str(trace_script)],
        check=True,
        capture_output=True,
        text=True,
        cwd=trace_script.parent,
        env={**os.environ, **environment},
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
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
    trace_script = Path(__file__).with_name("trace_contract.py").resolve()
    baseline_environment = dict(args.baseline_env)
    candidate_environment = dict(args.candidate_env)
    baseline = _run(args.baseline_python, trace_script, baseline_environment)
    candidate = _run(args.candidate_python, trace_script, candidate_environment)
    environment_passed = baseline["traces"] == candidate["traces"]
    preprocessing_passed = baseline["preprocessing"] == candidate["preprocessing"]
    passed = environment_passed and preprocessing_passed
    print(
        json.dumps(
            {
                "schema_version": 1,
                "passed": passed,
                "baseline_package_path": baseline["package_path"],
                "candidate_package_path": candidate["package_path"],
                "environment_trace_count": len(candidate["traces"]),
                "preprocessing_trace_count": len(candidate["preprocessing"]),
                "environment_passed": environment_passed,
                "preprocessing_passed": preprocessing_passed,
                "baseline_env": baseline_environment,
                "candidate_env": candidate_environment,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
