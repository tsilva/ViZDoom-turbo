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

### Augmented environments

Augmented variants use the `<base>-Plus-v<version>` naming convention.
`VizdoomBasic-Plus-v1` preserves the canonical Basic scenario while sampling
one target appearance and one coordinated wall/floor/ceiling texture set per
lane on every reset:

```python
env = VizdoomTurboVecEnv(
    "VizdoomBasic-Plus-v1",
    num_envs=16,
    enemy_variants={
        "target": [
            "original",
            "basalt-furnace-sentinel-v1",
            "verdigris-ram-hound-v1",
        ],
    },
    surface_variants={
        "texture_set": [
            "original",
            "polar-bunker-v1",
            "solar-shrine-v1",
            "verdant-ruin-v1",
        ],
    },
)
observations, infos = env.reset(seed=7)
print(infos["target_variant_id"])
print(infos["texture_set_variant_id"])
```

Each non-original texture-set choice changes all three room surfaces together,
so a lane cannot mix materials from different themes. Omitting either variant
mapping samples uniformly from every catalog default.

`VizdoomDefendLine-Plus-v1` preserves the canonical Defend the Line mechanics
while independently selecting one configured appearance for each enemy and
surface role in every vector lane on every reset:

```python
env = VizdoomTurboVecEnv(
    "VizdoomDefendLine-Plus-v1",
    num_envs=16,
    enemy_variants={
        "shooter": [
            "original",
            "basalt-furnace-sentinel-v1",
        ],
        "fighter": [
            "original",
            "verdigris-ram-hound-v1",
        ],
    },
    surface_variants={
        "wall": [
            "original",
            "basalt-blocks-v1",
            "steel-panels-v1",
        ],
        "floor": [
            "original",
            "dark-stone-v1",
        ],
        "ceiling": [
            "original",
            "industrial-grid-v1",
        ],
    },
)
observations, infos = env.reset(seed=7)
print(infos["shooter_variant_id"])
print(infos["fighter_variant_id"])
print(infos["wall_variant_id"])
print(infos["floor_variant_id"])
print(infos["ceiling_variant_id"])
```

Selection is uniform within each configured role, reproducible under
`reset(seed=...)`, and driven by separate role RNG streams so it does not consume
gameplay, no-op, sticky-action, or another role's randomness. Omit
`enemy_variants` and `surface_variants` to use every role's catalog defaults. An
`enemy_variants` sequence remains a shorthand for configuring the shooter role
only. Masked resets resample selected lanes only. `enemy_variant_roles` and
`surface_variant_roles`, their read-only two-dimensional active-index arrays,
and the role-keyed `active_enemy_variant_ids()` and
`active_surface_variant_ids()` mappings expose the current choices.

The surface catalog also exposes three image-generated, coordinated sets through
the immutable `surface_variant_themes` mapping: `polar-bunker-v1`,
`solar-shrine-v1`, and `verdant-ruin-v1`. Each maps to matching wall, floor, and
ceiling ids generated as individual material sources with shared theme
references. Role selection remains independent, so configure each role with the
corresponding singleton id when a reset must use one intact visual theme.

In a GradLab environment config, declare the same list under
`env_config.env_args.enemy_variants` or `env_config.env_args.surface_variants`.

Reusable source frames, Doom patch lumps, manifests, proofs, and provenance live
under `vizdoom_turbo/assets/enemy_variants/`. Seamless 64×64 PLAYPAL surface
tiles, tiled proofs, prompts, manifests, and provenance live under
`vizdoom_turbo/assets/surface_variants/`. The packaged Plus WADs are built from
the editable scenario sources with:

```bash
uv run python scripts/build_basic_plus.py \
  --acc /absolute/path/to/acc \
  --acc-include /absolute/path/to/acc-source

uv run python scripts/build_defend_line_plus.py \
  --acc /absolute/path/to/acc \
  --acc-include /absolute/path/to/acc-source
```

New generated surface sources can be normalized into a compatible tile with
`scripts/process_surface_variant.py`; its manifest gate verifies opacity,
palette membership, exact dimensions, and measured wrap seams. The default
pipeline center-crops one material source and downsamples it directly to 64×64,
repairing only wrap axes that exceed the seam threshold. Legacy generated grids
remain supported through `--grid-row` and `--grid-column`; source and processed
comparison grids live with the editable Defend the Line Plus scenario sources.

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
