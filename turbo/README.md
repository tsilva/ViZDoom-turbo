<div align="center">
  <strong>⚡ High-throughput ViZDoom rollouts, native-vector by design. ⚡</strong>
  <br /><br />
  <img src="https://raw.githubusercontent.com/tsilva/ViZDoom-turbo/turbo/logo.png" alt="ViZDoom-turbo" width="420" />
</div>

`vizdoom-turbo` is a Python library for reinforcement-learning researchers who need fast, parallel ViZDoom environments. It provides a Gymnasium vector environment that can be used directly or selected as an isolated environment provider in `rlab`.

Each vector lane owns an independent `DoomGame`. Lanes advance concurrently through ViZDoom's native API, while a bounded Rust worker pool applies max-pooling, crop, resize, grayscale conversion, frame-stack rotation, and final CHW/HWC layout in one GIL-free native call. Resize geometry and area-sampling tables are compiled once per environment instead of rebuilt per step.

## Install

Install the published package from PyPI:

```bash
uv add vizdoom-turbo
```

To work from source:

```bash
git clone git@github.com:tsilva/ViZDoom-turbo.git
cd ViZDoom-turbo/turbo
uv sync --all-extras
```

Run Python and project commands through `uv run`.

## Use

```python
import numpy as np
from vizdoom_turbo import VizdoomTurboVecEnv

env = VizdoomTurboVecEnv(
    "VizdoomBasic-v1",
    num_envs=16,
    num_threads=8,
    obs_resize=(84, 84),
    obs_grayscale=True,
    obs_layout="chw",
    frame_skip=4,
    frame_stack=4,
    use_restricted_actions="minimal",
)

try:
    observations, infos = env.reset(seed=7)
    actions = np.zeros(env.num_envs, dtype=np.int64)
    observations, rewards, terminated, truncated, infos = env.step(actions)

    done = terminated | truncated
    if np.any(done):
        observations, infos = env.reset(
            options={
                "reset_mask": done,
                "state_indices": np.zeros(env.num_envs, dtype=np.int32),
            }
        )
finally:
    env.close()
```

The package accepts canonical registered `Vizdoom...` Gymnasium IDs and
ViZDoom `.cfg` paths.

## Turbo Vector API v1

`VizdoomTurboVecEnv` implements the strict Turbo Vector API v1:

- `metadata["turbo_api_version"]` is `1`, and `metadata["render_modes"]`
  advertises `rgb_array`.
- Immutable `capabilities` and `signal_schema` declarations describe supported
  features and the dtype, shape, and reset/step availability of every signal.
- `buttons`, `action_mode`, `action_preset`, `action_table`,
  `action_meanings`, and `action_table_hash` expose the resolved action
  semantics without provider-specific probing.
- `state_catalog` is an immutable ordered tuple. Callers select reset states
  with an `int32` `state_indices` array and inspect the read-only active indices
  with `active_state_indices()`; state sampling and lane routing remain
  caller-owned.
- `observation_ownership` and `observation_buffer_depth` declare the exact
  lifetime of returned observations. `render_lane(index)` renders one lane,
  `get_images()` renders all lanes, and `render()` renders lane zero.

## Use with rlab

Install this distribution in the `rlab` runtime, then select its provider:

```yaml
environment:
  env_provider: vizdoom-turbo
  env_config:
    game: VizdoomBasic-v1
    state: default
    n_envs: 16
    env_args:
      num_threads: 8
      use_restricted_actions: minimal
      obs_grayscale: true
      obs_layout: chw
      frame_stack: 4
    preprocessing:
      frame_skip: 4
      max_pool_frames: true
      observation_size: 84
      obs_resize_algorithm: area
    task:
      id: identity
      action: {set: native}
      signals: {}
      events: {}
      termination: {}
      reward: {reward_mode: native}
```

## Commands

```bash
uv sync --all-extras                                      # install project and dev dependencies
uv run pytest -q                                          # run Python and live-environment tests
uv run ruff check .                                       # lint Python
cargo fmt --check                                         # check Rust formatting
cargo clippy --all-targets --all-features -- -D warnings  # lint Rust
uv build --wheel                                          # build the distributable wheel
uv run python benchmarks/benchmark_sps.py                 # measure the canonical SPS profile
```

For release-to-release checks, use `benchmarks/compare_contract.py` to compare deterministic traces and `benchmarks/compare_sps.py` for alternating paired SPS measurements. Both commands accept separate baseline and candidate Python interpreters so the released wheel and working tree stay isolated.

## Notes

- Python 3.14 is supported. Release wheels target macOS ARM64 and Linux x86-64. Source builds require Rust 1.85 or newer.
- ViZDoom 1.3.0 supplies built-in scenarios and Freedoom assets. Commercial Doom IWADs are not included; pass one with `rom_path` when required.
- Autoreset is disabled. Terminal lanes retain their final observation and must be selected explicitly with a masked reset.
- Preprocessing supports crop removal or masking, max-pooling, nearest/bilinear/area resize, grayscale or RGB, frame skip, frame stacking, and CHW or HWC layouts.
- The native vector path supports image observations, `rgb_array` rendering, and one player. Recording is not supported.

## Architecture

![vizdoom-turbo architecture](https://raw.githubusercontent.com/tsilva/ViZDoom-turbo/turbo/architecture.png)

## License

[MIT](LICENSE)
