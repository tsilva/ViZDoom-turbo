from __future__ import annotations

import inspect
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode
from vizdoom_turbo import VizDoomTurboVecEnv, VizdoomTurboVecEnv, scenario_buttons

SUPPORTED_SCENARIOS = (
    "basic",
    "basic_audio",
    "basic_notifications",
    "deadly_corridor",
    "deathmatch",
    "defend_the_center",
    "defend_the_line",
    "health_gathering",
    "health_gathering_supreme",
    "my_way_home",
    "predict_position",
    "take_cover",
)
REGISTERED_TURBO_GAMES = {
    "VizdoomBasic-Turbo-v0": "VizdoomBasic-v1",
    "VizdoomDeadlyCorridor-Turbo-v0": "VizdoomDeadlyCorridor-v1",
    "VizdoomDefendCenter-Turbo-v0": "VizdoomDefendCenter-v1",
    "VizdoomDefendLine-Turbo-v0": "VizdoomDefendLine-v1",
    "VizdoomHealthGathering-Turbo-v0": "VizdoomHealthGathering-v1",
    "VizdoomHealthGatheringSupreme-Turbo-v0": "VizdoomHealthGatheringSupreme-v1",
    "VizdoomMyWayHome-Turbo-v0": "VizdoomMyWayHome-v1",
    "VizdoomPredictPosition-Turbo-v0": "VizdoomPredictPosition-v1",
    "VizdoomTakeCover-Turbo-v0": "VizdoomTakeCover-v1",
}


def make_env(**overrides) -> VizdoomTurboVecEnv:
    options = {
        "game": "VizdoomBasic-v1",
        "num_envs": 2,
        "num_threads": 2,
        "use_restricted_actions": "minimal",
        "obs_resize": (32, 40),
        "obs_grayscale": True,
        "obs_layout": "chw",
        "frame_skip": 2,
        "frame_stack": 4,
        "maxpool_last_two": True,
        "info_filter": "all",
    }
    options.update(overrides)
    return VizdoomTurboVecEnv(**options)


def test_public_signature_matches_turbo_constructor_contract() -> None:
    parameters = inspect.signature(VizdoomTurboVecEnv).parameters
    expected = {
        "game",
        "state",
        "scenario",
        "info",
        "use_restricted_actions",
        "record",
        "players",
        "inttype",
        "obs_type",
        "render_mode",
        "num_envs",
        "num_threads",
        "rom_path",
        "obs_copy",
        "obs_resize",
        "obs_crop",
        "obs_crop_mode",
        "obs_crop_fill",
        "obs_grayscale",
        "obs_resize_algorithm",
        "obs_layout",
        "frame_skip",
        "frame_stack",
        "maxpool_last_two",
        "noop_reset_max",
        "use_fire_reset",
        "sticky_action_prob",
        "reward_clip",
        "info_filter",
        "state_catalog",
    }
    assert expected <= set(parameters)
    assert VizDoomTurboVecEnv is VizdoomTurboVecEnv
    assert issubclass(VizdoomTurboVecEnv, gym.vector.VectorEnv)
    assert VizdoomTurboVecEnv.metadata["autoreset_mode"] is AutoresetMode.DISABLED
    assert VizdoomTurboVecEnv.metadata["turbo_api_version"] == 1
    assert gym.spec("VizdoomBasic-Turbo-v0").vector_entry_point == (
        "vizdoom_turbo:VizdoomTurboVecEnv"
    )
    assert scenario_buttons("VizdoomBasic-v1") == (
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "ATTACK",
    )
    for registered_id, game in REGISTERED_TURBO_GAMES.items():
        spec = gym.spec(registered_id)
        assert spec.vector_entry_point == "vizdoom_turbo:VizdoomTurboVecEnv"
        assert spec.kwargs["game"] == game


def test_turbo_api_v1_capabilities_signals_ownership_and_rendering() -> None:
    env = make_env()
    try:
        env.reset(seed=19)
        assert env.observation_ownership == "safe_view"
        assert env.observation_buffer_depth == 2
        assert env.live_snapshots_deterministic is True
        assert env.capabilities["supported_action_modes"] == (
            "all",
            "filtered",
            "multi_discrete",
            "custom_discrete",
        )
        assert tuple(env.signal_schema) == tuple(env._info_keys)
        images = env.get_images()
        assert len(images) == env.num_envs
        assert all(image.dtype == np.uint8 and image.ndim == 3 for image in images)
        np.testing.assert_array_equal(env.render(), images[0])
    finally:
        env.close()


def test_legacy_reset_selector_names_are_rejected() -> None:
    env = make_env()
    try:
        with pytest.raises(ValueError, match="unsupported reset options"):
            env.reset(
                options={
                    "start_indices": np.zeros(env.num_envs, dtype=np.int32)
                }
            )
        with pytest.raises(ValueError, match="unsupported reset options"):
            env.reset(options={"start_ids": np.full(env.num_envs, "default")})
    finally:
        env.close()


@pytest.mark.parametrize("scenario", SUPPORTED_SCENARIOS)
def test_every_supported_scenario_resets_and_steps(scenario: str) -> None:
    env = VizdoomTurboVecEnv(
        game=scenario,
        num_envs=1,
        num_threads=1,
        use_restricted_actions="minimal",
        obs_resize=(24, 32),
        obs_grayscale=True,
        obs_layout="chw",
        frame_skip=1,
        frame_stack=2,
        info_filter="none",
    )
    try:
        observations, infos = env.reset(seed=918)
        assert observations.shape == (1, 2, 24, 32)
        assert infos["state_index"].tolist() == [0]
        transition = env.step(np.zeros(1, dtype=np.int64))
        assert transition[0].shape == observations.shape
        assert transition[1].shape == (1,)
    finally:
        env.close()


def test_real_vector_step_and_masked_reset_preserve_other_lane() -> None:
    env = make_env()
    try:
        assert all(
            Path(game.get_doom_config_path()).parent == Path(env._tempdir.name)
            for game in env._games
        )
        observations, infos = env.reset(seed=123)
        assert observations.shape == (2, 4, 32, 40)
        assert observations.dtype == np.uint8
        assert env.observation_space.contains(observations)
        assert infos["state_index"].tolist() == [0, 0]
        assert infos["_state_index"].tolist() == [True, True]

        observations, rewards, terminated, truncated, infos = env.step(
            np.asarray([0, 1], dtype=np.int64)
        )
        assert rewards.shape == terminated.shape == truncated.shape == (2,)
        assert infos["episode_time"].shape == (2,)
        lane_one_observation = observations[1].copy()
        lane_one_raw = env.render_lane(1)
        lane_one_time = infos["episode_time"][1]

        mask = np.asarray([True, False], dtype=np.bool_)
        state_indices = np.zeros(2, dtype=np.int32)
        observations, infos = env.reset(
            seed=[999, None],
            options={"reset_mask": mask, "state_indices": state_indices},
        )

        np.testing.assert_array_equal(observations[1], lane_one_observation)
        np.testing.assert_array_equal(env.render_lane(1), lane_one_raw)
        assert infos["_state_index"].tolist() == [True, False]
        assert env._games[1].get_episode_time() == lane_one_time
    finally:
        env.close()


def test_snapshot_restore_replays_identical_transition() -> None:
    env = make_env(sticky_action_prob=0.25)
    try:
        env.reset(seed=777)
        env.step(np.asarray([1, 2], dtype=np.int64))
        mask = np.ones(2, dtype=np.bool_)
        snapshots = env.capture_snapshots(mask)
        actions = np.asarray([2, 3], dtype=np.int64)
        raw_expected = env.step(actions)
        expected = (
            *(value.copy() for value in raw_expected[:4]),
            {key: value.copy() for key, value in raw_expected[4].items()},
        )

        state_indices = np.full(2, -1, dtype=np.int32)
        restored, infos = env.reset(
            options={
                "reset_mask": mask,
                "state_indices": state_indices,
                "snapshots": snapshots,
            }
        )
        assert infos["start_source"].tolist() == ["snapshot", "snapshot"]
        assert restored.shape == expected[0].shape
        actual = env.step(actions)

        for expected_array, actual_array in zip(expected[:4], actual[:4], strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)
        for key in expected[4]:
            np.testing.assert_array_equal(actual[4][key], expected[4][key])
    finally:
        env.close()


def test_safe_view_survives_one_environment_call() -> None:
    env = make_env(obs_copy="safe_view")
    try:
        first, _infos = env.reset(seed=3)
        first_owned_value = first.copy()
        env.step(np.asarray([1, 1], dtype=np.int64))
        np.testing.assert_array_equal(first, first_owned_value)
    finally:
        env.close()


def test_terminal_lane_blocks_step_until_masked_reset() -> None:
    env = make_env(
        frame_skip=8,
        maxpool_last_two=False,
        vizdoom_config={"episode_timeout": 20, "episode_start_time": 1},
    )
    try:
        env.reset(seed=5)
        done = np.zeros(2, dtype=np.bool_)
        for _ in range(8):
            _obs, _reward, terminated, truncated, _infos = env.step(
                np.zeros(2, dtype=np.int64)
            )
            done = terminated | truncated
            if np.any(done):
                break
        assert np.all(done)
        with pytest.raises(RuntimeError, match="terminal lanes must be reset"):
            env.step(np.zeros(2, dtype=np.int64))
        env.reset(
            options={
                "reset_mask": done,
                "state_indices": np.zeros(2, dtype=np.int32),
            }
        )
        env.step(np.zeros(2, dtype=np.int64))
    finally:
        env.close()


def test_custom_action_table_is_exact_and_hashed() -> None:
    table = [[], ["MOVE_LEFT"], ["MOVE_RIGHT", "ATTACK"]]
    env = make_env(use_restricted_actions=table)
    try:
        assert env.action_mode == "custom_discrete"
        assert env.action_table == ((), ("MOVE_LEFT",), ("MOVE_RIGHT", "ATTACK"))
        assert env.action_meanings == ("noop", "move_left", "move_right_attack")
        assert len(env.action_table_hash) == 64
        np.testing.assert_array_equal(
            env._native_actions(np.asarray([1, 2])),
            [[1, 0, 0], [0, 1, 1]],
        )
    finally:
        env.close()
