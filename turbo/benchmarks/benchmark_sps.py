#!/usr/bin/env python3
"""Measure training-loop environment steps per second for vizdoom-turbo."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import vizdoom_turbo
from vizdoom_turbo import VizdoomTurboVecEnv


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _lane_counts(value: str) -> tuple[int, ...]:
    counts = tuple(_positive_int(item.strip()) for item in value.split(","))
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("num-envs values must be unique")
    return counts


def _run_steps(env: VizdoomTurboVecEnv, actions: np.ndarray, count: int) -> None:
    for _ in range(count):
        _observations, _rewards, terminated, truncated, _infos = env.step(actions)
        reset_mask = terminated | truncated
        if np.any(reset_mask):
            state_indices = np.full(env.num_envs, -1, dtype=np.int32)
            state_indices[reset_mask] = 0
            env.reset(
                options={
                    "reset_mask": reset_mask,
                    "state_indices": state_indices,
                }
            )


def _measure(
    num_envs: int,
    *,
    warmup_steps: int,
    measured_steps: int,
    samples: int,
) -> list[float]:
    env = VizdoomTurboVecEnv(
        game="VizdoomBasic-v1",
        num_envs=num_envs,
        num_threads=num_envs,
        use_restricted_actions="minimal",
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_layout="chw",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        info_filter="none",
        obs_copy="unsafe_view",
    )
    try:
        env.reset(seed=123)
        actions = np.arange(num_envs, dtype=np.int64) % env.single_action_space.n
        _run_steps(env, actions, warmup_steps)
        values = []
        for _ in range(samples):
            started = time.perf_counter()
            _run_steps(env, actions, measured_steps)
            elapsed = time.perf_counter() - started
            values.append(num_envs * measured_steps / elapsed)
        return values
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=_lane_counts, default=(1, 2, 4, 8, 16))
    parser.add_argument("--warmup-steps", type=_positive_int, default=40)
    parser.add_argument("--measured-steps", type=_positive_int, default=250)
    parser.add_argument("--samples", type=_positive_int, default=7)
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args()

    try:
        package_version = version("vizdoom-turbo")
    except PackageNotFoundError:
        package_version = "unknown"
    runs = {}
    for num_envs in args.num_envs:
        values = _measure(
            num_envs,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            samples=args.samples,
        )
        runs[str(num_envs)] = {
            "samples": values,
            "median_sps": statistics.median(values),
        }
    payload = {
        "schema_version": 1,
        "label": args.label,
        "package_version": package_version,
        "package_path": str(Path(vizdoom_turbo.__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "profile": {
            "game": "VizdoomBasic-v1",
            "observation": "uint8 84x84 grayscale CHW stack=4",
            "frame_skip": 4,
            "maxpool_last_two": True,
            "info_filter": "none",
            "obs_copy": "unsafe_view",
            "manual_terminal_reset": True,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
        },
        "runs": runs,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
