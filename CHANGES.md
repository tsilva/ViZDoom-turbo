# Changelog

## Unreleased

- Nothing yet.

## 1.3.0.post2 - 2026-07-27

- Add the immutable Turbo Vector API v1 declaration for capabilities, signals,
  action semantics, observation ownership, state catalogs, and per-lane RGB
  rendering.
- Remove legacy state-name properties and reset selectors in favor of
  `state_catalog`, `active_state_indices()`, and `state_indices`.

## 1.3.0.post1 - 2026-07-27

- Align turbo release versions with upstream ViZDoom using PEP 440 post releases.

## 0.1.3 - 2026-07-27

- Optimize native vector preprocessing throughput.

## 0.1.2 - 2026-07-26

- Move `vizdoom-turbo` into the `turbo` branch of the ViZDoom fork.
- Namespace package release tags separately from upstream ViZDoom tags.
- Exclude generated Python cache files and build artifacts from release distributions.

## 0.1.1 - 2026-07-26

- Install ViZDoom's source-build dependencies for Intel macOS release wheels.
- Refresh the Rust lockfile while preparing version bumps.

## 0.1.0 - 2026-07-26

- Add native-vector Gymnasium environments with concurrent ViZDoom lanes,
  deterministic masked resets, and batched Rust preprocessing.
