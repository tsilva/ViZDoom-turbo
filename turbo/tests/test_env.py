from __future__ import annotations

import inspect
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import vizdoom as vzd
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
    "VizdoomBasic-Plus-v1": "VizdoomBasic-Plus-v1",
    "VizdoomDeadlyCorridor-Turbo-v0": "VizdoomDeadlyCorridor-v1",
    "VizdoomDefendCenter-Turbo-v0": "VizdoomDefendCenter-v1",
    "VizdoomDefendLine-Turbo-v0": "VizdoomDefendLine-v1",
    "VizdoomDefendLine-Plus-v1": "VizdoomDefendLine-Plus-v1",
    "VizdoomHealthGathering-Turbo-v0": "VizdoomHealthGathering-v1",
    "VizdoomHealthGatheringSupreme-Turbo-v0": "VizdoomHealthGatheringSupreme-v1",
    "VizdoomMyWayHome-Turbo-v0": "VizdoomMyWayHome-v1",
    "VizdoomPredictPosition-Turbo-v0": "VizdoomPredictPosition-v1",
    "VizdoomTakeCover-Turbo-v0": "VizdoomTakeCover-v1",
}
APPEARANCE_VARIANT_RESET_INFO_KEYS = {
    f"{prefix}{role}_variant_{suffix}"
    for role in (
        "target",
        "texture_set",
        "shooter",
        "fighter",
        "wall",
        "floor",
        "ceiling",
    )
    for prefix in ("", "_")
    for suffix in ("index", "id")
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


def make_exact_env(**overrides) -> VizdoomTurboVecEnv:
    options = {
        "game": "VizdoomBasic-v1",
        "num_envs": 4,
        "num_threads": 4,
        "use_restricted_actions": "discrete",
        "obs_copy": "safe_view",
        "obs_resize": (84, 84),
        "obs_grayscale": True,
        "obs_layout": "chw",
        "frame_stack": 4,
        "frame_skip": 4,
        "maxpool_last_two": False,
        "sticky_action_prob": 0,
        "obs_resize_algorithm": "area",
        "info_filter": {"mode": "all", "keys": ["killcount"]},
        "game_variables": ["KILLCOUNT"],
    }
    options.update(overrides)
    return VizdoomTurboVecEnv(**options)


def assert_info_equal(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def assert_mechanical_info_equal(
    actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> None:
    actual_keys = set(actual) - APPEARANCE_VARIANT_RESET_INFO_KEYS
    expected_keys = set(expected) - APPEARANCE_VARIANT_RESET_INFO_KEYS
    assert actual_keys == expected_keys
    for key in actual_keys:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def test_public_signature_matches_turbo_constructor_contract() -> None:
    parameters = inspect.signature(VizdoomTurboVecEnv).parameters
    assert parameters["use_fire_reset"].default is False
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
        "enemy_variants",
        "surface_variants",
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
    assert scenario_buttons("VizdoomBasic-Plus-v1") == (
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "ATTACK",
    )
    assert scenario_buttons("VizdoomDefendLine-Plus-v1") == (
        "TURN_LEFT",
        "TURN_RIGHT",
        "ATTACK",
    )
    for registered_id, game in REGISTERED_TURBO_GAMES.items():
        spec = gym.spec(registered_id)
        assert spec.vector_entry_point == "vizdoom_turbo:VizdoomTurboVecEnv"
        assert spec.kwargs["game"] == game


def test_removed_state_dir_is_rejected() -> None:
    assert "state_dir" not in inspect.signature(VizdoomTurboVecEnv).parameters
    with pytest.raises(TypeError, match="state_dir"):
        VizdoomTurboVecEnv(state_dir="/tmp/states")


def test_basic_plus_samples_coherent_texture_sets_and_targets() -> None:
    common = {
        "game": "VizdoomBasic-Plus-v1",
        "num_envs": 16,
        "num_threads": 4,
    }
    left = make_exact_env(**common)
    right = make_exact_env(**common)
    try:
        left_observations, left_infos = left.reset(seed=0)
        right_observations, right_infos = right.reset(seed=0)
        np.testing.assert_array_equal(left_observations, right_observations)
        assert_info_equal(left_infos, right_infos)
        assert left.enemy_variant_roles == ("target",)
        assert left.surface_variant_roles == ("texture_set",)
        assert dict(left.enemy_variants) == {
            "target": (
                "original",
                "basalt-furnace-sentinel-v1",
                "verdigris-ram-hound-v1",
            )
        }
        assert dict(left.surface_variants) == {
            "texture_set": (
                "original",
                "polar-bunker-v1",
                "solar-shrine-v1",
                "verdant-ruin-v1",
            )
        }
        assert set(left.active_enemy_variant_ids()["target"]) == set(
            left.enemy_variants["target"]
        )
        assert set(left.active_surface_variant_ids()["texture_set"]) == set(
            left.surface_variants["texture_set"]
        )
        assert left.active_enemy_variant_ids() == right.active_enemy_variant_ids()
        assert left.active_surface_variant_ids() == right.active_surface_variant_ids()
        assert left.capabilities["supports_enemy_variants"] is True
        assert left.capabilities["supports_surface_variants"] is True
        assert len(left.enemy_variant_wad_sha256) == 64
        assert left.surface_variant_wad_sha256 == left.enemy_variant_wad_sha256

        before_enemy_ids = left.active_enemy_variant_ids()
        before_surface_ids = left.active_surface_variant_ids()
        mask = np.zeros(left.num_envs, dtype=np.bool_)
        mask[0] = True
        _observations, masked_infos = left.reset(
            seed=[2, *([None] * (left.num_envs - 1))],
            options={"reset_mask": mask},
        )
        assert left.active_enemy_variant_ids()["target"][1:] == before_enemy_ids["target"][1:]
        assert (
            left.active_surface_variant_ids()["texture_set"][1:]
            == before_surface_ids["texture_set"][1:]
        )
        assert masked_infos["_target_variant_id"].tolist() == mask.tolist()
        assert masked_infos["_texture_set_variant_id"].tolist() == mask.tolist()
    finally:
        left.close()
        right.close()


def test_basic_plus_original_appearance_is_mechanically_exact() -> None:
    common = {"num_envs": 2, "num_threads": 2}
    canonical = make_exact_env(game="VizdoomBasic-v1", **common)
    plus = make_exact_env(
        game="VizdoomBasic-Plus-v1",
        enemy_variants={"target": ["original"]},
        surface_variants={"texture_set": ["original"]},
        **common,
    )
    try:
        canonical_observations, canonical_infos = canonical.reset(seed=727)
        plus_observations, plus_infos = plus.reset(seed=727)
        np.testing.assert_array_equal(plus_observations, canonical_observations)
        assert_mechanical_info_equal(plus_infos, canonical_infos)

        actions = np.zeros(2, dtype=np.int64)
        for _ in range(8):
            canonical_transition = canonical.step(actions)
            plus_transition = plus.step(actions)
            for actual, expected in zip(
                plus_transition[:4],
                canonical_transition[:4],
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            assert_info_equal(plus_transition[4], canonical_transition[4])
    finally:
        canonical.close()
        plus.close()


@pytest.mark.parametrize(
    ("enemy_id", "texture_set_id"),
    (
        ("basalt-furnace-sentinel-v1", "original"),
        ("verdigris-ram-hound-v1", "original"),
        ("original", "polar-bunker-v1"),
        ("original", "solar-shrine-v1"),
        ("original", "verdant-ruin-v1"),
    ),
)
def test_basic_plus_variants_change_pixels(
    enemy_id: str,
    texture_set_id: str,
) -> None:
    common = {
        "game": "VizdoomBasic-Plus-v1",
        "num_envs": 2,
        "num_threads": 2,
        "frame_stack": 1,
        "frame_skip": 1,
        "obs_grayscale": False,
        "obs_layout": "hwc",
    }
    original = make_exact_env(
        enemy_variants={"target": ["original"]},
        surface_variants={"texture_set": ["original"]},
        **common,
    )
    variant = make_exact_env(
        enemy_variants={"target": [enemy_id]},
        surface_variants={"texture_set": [texture_set_id]},
        **common,
    )
    try:
        original_observations, _original_infos = original.reset(seed=613)
        variant_observations, variant_infos = variant.reset(seed=613)
        assert np.any(variant_observations != original_observations)
        assert variant_infos["target_variant_id"].tolist() == [enemy_id, enemy_id]
        assert variant_infos["texture_set_variant_id"].tolist() == [
            texture_set_id,
            texture_set_id,
        ]
        actions = np.zeros(2, dtype=np.int64)
        for _ in range(8):
            original_transition = original.step(actions)
            variant_transition = variant.step(actions)
            for actual, expected in zip(
                variant_transition[1:4],
                original_transition[1:4],
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            assert_mechanical_info_equal(variant_transition[4], original_transition[4])
    finally:
        original.close()
        variant.close()


def test_defend_line_plus_catalog_is_explicit_and_rejects_invalid_use() -> None:
    env = make_exact_env(
        game="VizdoomDefendLine-Plus-v1",
        num_envs=1,
        num_threads=1,
    )
    try:
        assert env.enemy_variant_roles == ("shooter", "fighter")
        assert dict(env.enemy_variants) == {
            "shooter": ("original", "basalt-furnace-sentinel-v1"),
            "fighter": ("original", "verdigris-ram-hound-v1"),
        }
        assert len(env.enemy_variant_catalog_sha256) == 64
        assert len(env.enemy_variant_wad_sha256) == 64
        assert env.capabilities["supports_enemy_variants"] is True
        assert env.surface_variant_roles == ("wall", "floor", "ceiling")
        assert dict(env.surface_variants) == {
            "wall": (
                "original",
                "basalt-blocks-v1",
                "steel-panels-v1",
                "polar-bunker-wall-v1",
                "solar-shrine-wall-v1",
                "verdant-ruin-wall-v1",
            ),
            "floor": (
                "original",
                "dark-stone-v1",
                "polar-bunker-floor-v1",
                "solar-shrine-floor-v1",
                "verdant-ruin-floor-v1",
            ),
            "ceiling": (
                "original",
                "industrial-grid-v1",
                "polar-bunker-ceiling-v1",
                "solar-shrine-ceiling-v1",
                "verdant-ruin-ceiling-v1",
            ),
        }
        assert {
            theme: dict(variants) for theme, variants in env.surface_variant_themes.items()
        } == {
            "polar-bunker-v1": {
                "wall": "polar-bunker-wall-v1",
                "floor": "polar-bunker-floor-v1",
                "ceiling": "polar-bunker-ceiling-v1",
            },
            "solar-shrine-v1": {
                "wall": "solar-shrine-wall-v1",
                "floor": "solar-shrine-floor-v1",
                "ceiling": "solar-shrine-ceiling-v1",
            },
            "verdant-ruin-v1": {
                "wall": "verdant-ruin-wall-v1",
                "floor": "verdant-ruin-floor-v1",
                "ceiling": "verdant-ruin-ceiling-v1",
            },
        }
        assert len(env.surface_variant_catalog_sha256) == 64
        assert env.surface_variant_wad_sha256 == env.enemy_variant_wad_sha256
        assert env.capabilities["supports_surface_variants"] is True
    finally:
        env.close()

    with pytest.raises(ValueError, match="unknown shooter variant"):
        make_exact_env(
            game="VizdoomDefendLine-Plus-v1",
            num_envs=1,
            num_threads=1,
            enemy_variants=["missing"],
        )
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        make_exact_env(
            game="VizdoomDefendLine-Plus-v1",
            num_envs=1,
            num_threads=1,
            enemy_variants=["original", "original"],
        )
    with pytest.raises(ValueError, match="unknown Defend the Line enemy role"):
        make_exact_env(
            game="VizdoomDefendLine-Plus-v1",
            num_envs=1,
            num_threads=1,
            enemy_variants={"missing": ["original"]},
        )
    with pytest.raises(ValueError, match="only supported"):
        make_exact_env(
            num_envs=1,
            num_threads=1,
            enemy_variants=["original"],
        )
    with pytest.raises(ValueError, match="unknown wall surface variant"):
        make_exact_env(
            game="VizdoomDefendLine-Plus-v1",
            num_envs=1,
            num_threads=1,
            surface_variants={"wall": ["missing"]},
        )
    with pytest.raises(ValueError, match="unknown Defend the Line surface role"):
        make_exact_env(
            game="VizdoomDefendLine-Plus-v1",
            num_envs=1,
            num_threads=1,
            surface_variants={"missing": ["original"]},
        )
    with pytest.raises(ValueError, match="only supported"):
        make_exact_env(
            num_envs=1,
            num_threads=1,
            surface_variants={"wall": ["original"]},
        )


def test_defend_line_plus_original_variant_is_mechanically_exact() -> None:
    common = {
        "num_envs": 2,
        "num_threads": 2,
    }
    canonical = make_exact_env(game="VizdoomDefendLine-v1", **common)
    plus = make_exact_env(
        game="VizdoomDefendLine-Plus-v1",
        enemy_variants={
            "shooter": ["original"],
            "fighter": ["original"],
        },
        surface_variants={
            "wall": ["original"],
            "floor": ["original"],
            "ceiling": ["original"],
        },
        **common,
    )
    try:
        canonical_observations, canonical_infos = canonical.reset(seed=727)
        plus_observations, plus_infos = plus.reset(seed=727)
        np.testing.assert_array_equal(plus_observations, canonical_observations)
        assert_mechanical_info_equal(plus_infos, canonical_infos)
        assert dict(plus.active_enemy_variant_ids()) == {
            "shooter": ("original", "original"),
            "fighter": ("original", "original"),
        }
        assert plus_infos["shooter_variant_id"].tolist() == ["original", "original"]
        assert plus_infos["fighter_variant_id"].tolist() == ["original", "original"]
        assert dict(plus.active_surface_variant_ids()) == {
            "wall": ("original", "original"),
            "floor": ("original", "original"),
            "ceiling": ("original", "original"),
        }
        assert plus._config_hash != canonical._config_hash

        actions = np.zeros(2, dtype=np.int64)
        for _ in range(8):
            canonical_transition = canonical.step(actions)
            plus_transition = plus.step(actions)
            for actual, expected in zip(
                plus_transition[:4],
                canonical_transition[:4],
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            assert_info_equal(plus_transition[4], canonical_transition[4])
    finally:
        canonical.close()
        plus.close()


def test_defend_line_plus_basalt_changes_pixels_not_transition_signals() -> None:
    common = {
        "game": "VizdoomDefendLine-Plus-v1",
        "num_envs": 2,
        "num_threads": 2,
    }
    original = make_exact_env(
        enemy_variants={
            "shooter": ["original"],
            "fighter": ["original"],
        },
        **common,
    )
    basalt = make_exact_env(
        enemy_variants={
            "shooter": ["basalt-furnace-sentinel-v1"],
            "fighter": ["original"],
        },
        **common,
    )
    try:
        original_observations, original_infos = original.reset(seed=311)
        basalt_observations, basalt_infos = basalt.reset(seed=311)
        assert np.any(basalt_observations != original_observations)
        assert_mechanical_info_equal(basalt_infos, original_infos)
        assert basalt.active_enemy_variant_ids()["shooter"] == (
            "basalt-furnace-sentinel-v1",
            "basalt-furnace-sentinel-v1",
        )

        actions = np.zeros(2, dtype=np.int64)
        saw_visual_difference = True
        for _ in range(8):
            original_transition = original.step(actions)
            basalt_transition = basalt.step(actions)
            saw_visual_difference = saw_visual_difference or bool(
                np.any(basalt_transition[0] != original_transition[0])
            )
            for actual, expected in zip(
                basalt_transition[1:4],
                original_transition[1:4],
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            assert_info_equal(basalt_transition[4], original_transition[4])
        assert saw_visual_difference
    finally:
        original.close()
        basalt.close()


def test_defend_line_plus_melee_variant_changes_pixels_not_mechanics() -> None:
    common = {
        "game": "VizdoomDefendLine-Plus-v1",
        "num_envs": 2,
        "num_threads": 2,
        "enemy_variants": {"shooter": ["original"]},
    }
    original = make_exact_env(
        enemy_variants={
            "shooter": ["original"],
            "fighter": ["original"],
        },
        game=common["game"],
        num_envs=common["num_envs"],
        num_threads=common["num_threads"],
    )
    verdigris = make_exact_env(
        enemy_variants={
            "shooter": ["original"],
            "fighter": ["verdigris-ram-hound-v1"],
        },
        game=common["game"],
        num_envs=common["num_envs"],
        num_threads=common["num_threads"],
    )
    try:
        original_observations, original_infos = original.reset(seed=419)
        variant_observations, variant_infos = verdigris.reset(seed=419)
        assert np.any(variant_observations != original_observations)
        assert_mechanical_info_equal(variant_infos, original_infos)
        assert verdigris.active_enemy_variant_ids()["fighter"] == (
            "verdigris-ram-hound-v1",
            "verdigris-ram-hound-v1",
        )

        actions = np.zeros(2, dtype=np.int64)
        for _ in range(12):
            original_transition = original.step(actions)
            variant_transition = verdigris.step(actions)
            for actual, expected in zip(
                variant_transition[1:4],
                original_transition[1:4],
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            assert_info_equal(variant_transition[4], original_transition[4])
    finally:
        original.close()
        verdigris.close()


@pytest.mark.parametrize(
    ("role", "variant_id"),
    (
        ("wall", "basalt-blocks-v1"),
        ("wall", "steel-panels-v1"),
        ("wall", "polar-bunker-wall-v1"),
        ("wall", "solar-shrine-wall-v1"),
        ("wall", "verdant-ruin-wall-v1"),
        ("floor", "dark-stone-v1"),
        ("floor", "polar-bunker-floor-v1"),
        ("floor", "solar-shrine-floor-v1"),
        ("floor", "verdant-ruin-floor-v1"),
        ("ceiling", "industrial-grid-v1"),
        ("ceiling", "polar-bunker-ceiling-v1"),
        ("ceiling", "solar-shrine-ceiling-v1"),
        ("ceiling", "verdant-ruin-ceiling-v1"),
    ),
)
def test_defend_line_plus_surface_changes_pixels_not_mechanics(role: str, variant_id: str) -> None:
    common = {
        "game": "VizdoomDefendLine-Plus-v1",
        "num_envs": 2,
        "num_threads": 2,
        "enemy_variants": {
            "shooter": ["original"],
            "fighter": ["original"],
        },
    }
    original_surfaces = {
        "wall": ["original"],
        "floor": ["original"],
        "ceiling": ["original"],
    }
    selected_surfaces = dict(original_surfaces)
    selected_surfaces[role] = [variant_id]
    original = make_exact_env(surface_variants=original_surfaces, **common)
    variant = make_exact_env(surface_variants=selected_surfaces, **common)
    try:
        original_observations, original_infos = original.reset(seed=613)
        variant_observations, variant_infos = variant.reset(seed=613)
        assert np.any(variant_observations != original_observations)
        assert_mechanical_info_equal(variant_infos, original_infos)
        assert variant.active_surface_variant_ids()[role] == (
            variant_id,
            variant_id,
        )

        actions = np.zeros(2, dtype=np.int64)
        for _ in range(8):
            original_transition = original.step(actions)
            variant_transition = variant.step(actions)
            for actual, expected in zip(
                variant_transition[1:4],
                original_transition[1:4],
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            assert_info_equal(variant_transition[4], original_transition[4])
    finally:
        original.close()
        variant.close()


def test_defend_line_plus_default_mix_is_seed_reproducible() -> None:
    common = {
        "game": "VizdoomDefendLine-Plus-v1",
        "num_envs": 16,
        "num_threads": 4,
    }
    left = make_exact_env(**common)
    right = make_exact_env(**common)
    try:
        left_observations, left_infos = left.reset(seed=0)
        right_observations, right_infos = right.reset(seed=0)
        np.testing.assert_array_equal(left_observations, right_observations)
        assert_info_equal(left_infos, right_infos)
        assert left.active_enemy_variant_ids() == right.active_enemy_variant_ids()
        assert left.active_surface_variant_ids() == right.active_surface_variant_ids()
        active_ids = left.active_enemy_variant_ids()
        assert set(active_ids["shooter"]) == {
            "original",
            "basalt-furnace-sentinel-v1",
        }
        assert set(active_ids["fighter"]) == {
            "original",
            "verdigris-ram-hound-v1",
        }
        active_surface_ids = left.active_surface_variant_ids()
        for role, ids in active_surface_ids.items():
            assert 1 < len(set(ids))
            assert set(ids) <= set(left.surface_variants[role])
        actions = np.arange(16, dtype=np.int64) % 4
        for _ in range(8):
            left_transition = left.step(actions)
            right_transition = right.step(actions)
            for actual, expected in zip(
                left_transition[:4],
                right_transition[:4],
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            assert_info_equal(left_transition[4], right_transition[4])

        before_ids = {role: values for role, values in left.active_enemy_variant_ids().items()}
        before_surface_ids = {
            role: values for role, values in left.active_surface_variant_ids().items()
        }
        before_unmasked = left_transition[0][1:].copy()
        mask = np.zeros(16, dtype=np.bool_)
        mask[0] = True
        left_observations, left_infos = left.reset(
            seed=[2, *([None] * 15)],
            options={"reset_mask": mask},
        )
        after_ids = left.active_enemy_variant_ids()
        assert after_ids["shooter"][0] == "original"
        assert after_ids["fighter"][0] == "verdigris-ram-hound-v1"
        for role in left.enemy_variant_roles:
            assert after_ids[role][1:] == before_ids[role][1:]
        after_surface_ids = left.active_surface_variant_ids()
        for role in left.surface_variant_roles:
            assert after_surface_ids[role][1:] == before_surface_ids[role][1:]
        np.testing.assert_array_equal(left_observations[1:], before_unmasked)
        assert left_infos["_shooter_variant_id"].tolist() == mask.tolist()
        assert left_infos["_fighter_variant_id"].tolist() == mask.tolist()
        assert left_infos["_wall_variant_id"].tolist() == mask.tolist()
        assert left_infos["_floor_variant_id"].tolist() == mask.tolist()
        assert left_infos["_ceiling_variant_id"].tolist() == mask.tolist()
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize(
    "game",
    ("VizdoomBasic-Plus-v1", "VizdoomDefendLine-Plus-v1"),
)
def test_plus_snapshot_restore_preserves_appearance(game: str) -> None:
    env = make_exact_env(
        game=game,
        num_envs=2,
        num_threads=2,
    )
    try:
        env.reset(seed=881)
        env.step(np.asarray([0, 1], dtype=np.int64))
        enemy_ids = dict(env.active_enemy_variant_ids())
        surface_ids = dict(env.active_surface_variant_ids())
        mask = np.ones(2, dtype=np.bool_)
        snapshots = env.capture_snapshots(mask)
        actions = np.asarray([2, 3], dtype=np.int64)
        expected = env.step(actions)

        env.reset(
            options={
                "reset_mask": mask,
                "state_indices": np.full(2, -1, dtype=np.int32),
                "snapshots": snapshots,
            }
        )
        assert dict(env.active_enemy_variant_ids()) == enemy_ids
        assert dict(env.active_surface_variant_ids()) == surface_ids
        actual = env.step(actions)
        for actual_array, expected_array in zip(actual[:4], expected[:4], strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)
        assert_info_equal(actual[4], expected[4])
    finally:
        env.close()


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
            env.reset(options={"start_indices": np.zeros(env.num_envs, dtype=np.int32)})
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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux user-directory path")
def test_parallel_startup_tolerates_shared_user_directory_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for attempt in range(8):
        working_directory = tmp_path / str(attempt)
        working_directory.mkdir()
        monkeypatch.chdir(working_directory)
        env = make_env()
        try:
            observations, _infos = env.reset(seed=attempt)
            assert observations.shape == (2, 4, 32, 40)
            assert (working_directory / "_vizdoom").is_dir()
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
            _obs, _reward, terminated, truncated, _infos = env.step(np.zeros(2, dtype=np.int64))
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
        out = np.empty((2, 3), dtype=np.float64)
        assert env._native_actions(np.asarray([2, 1]), out=out) is out
        np.testing.assert_array_equal(out, [[0, 1, 1], [1, 0, 0]])
    finally:
        env.close()


def test_exact_profile_detects_custom_native_core() -> None:
    env = make_exact_env(num_envs=1, num_threads=1)
    try:
        observations, _infos = env.reset(seed=31)
        assert hasattr(vzd, "_TurboBatchStepper")
        assert env._native_stepper is not None
        assert env._use_indexed_native is True
        assert env._games[0].get_screen_format() == vzd.ScreenFormat.DOOM_256_COLORS8
        assert observations.shape == (1, 4, 84, 84)
    finally:
        env.close()


@pytest.mark.parametrize("crop_mode", ["remove", "mask"])
def test_enabled_hud_crop_runs_on_native_indexed_path(
    monkeypatch: pytest.MonkeyPatch,
    crop_mode: str,
) -> None:
    options = {
        "num_envs": 1,
        "num_threads": 1,
        "vizdoom_config": {"render_hud": True},
        "obs_crop": (0, 32, 0, 0),
        "obs_crop_mode": crop_mode,
        "obs_crop_fill": 0,
    }
    native = make_exact_env(**options)
    monkeypatch.setenv("VIZDOOM_TURBO_DISABLE_NATIVE_PIPELINE", "1")
    fallback = make_exact_env(**options)
    try:
        native_observations, native_infos = native.reset(seed=31)
        fallback_observations, fallback_infos = fallback.reset(seed=31)

        assert native._use_indexed_native is True
        assert native._native_stepper is not None
        assert native._games[0].get_screen_format() == vzd.ScreenFormat.DOOM_256_COLORS8
        assert fallback._use_indexed_native is False
        assert fallback._native_stepper is None
        np.testing.assert_array_equal(native_observations, fallback_observations)
        assert_info_equal(native_infos, fallback_infos)
        if crop_mode == "mask":
            assert np.all(native_observations[:, :, 73:, :] == 0)
            assert np.any(native_observations[:, :, :73, :] != 0)
        else:
            assert np.any(native_observations != 0)

        actions = np.zeros(1, dtype=np.int64)
        native_transition = native.step(actions)
        fallback_transition = fallback.step(actions)
        for native_value, fallback_value in zip(
            native_transition[:4], fallback_transition[:4], strict=True
        ):
            np.testing.assert_array_equal(native_value, fallback_value)
        assert_info_equal(native_transition[4], fallback_transition[4])
    finally:
        native.close()
        fallback.close()


@pytest.mark.parametrize("frame_skip", [0, -1])
def test_nonpositive_frame_skip_is_rejected(frame_skip: int) -> None:
    with pytest.raises(ValueError, match="frame_skip must be a positive integer"):
        make_exact_env(frame_skip=frame_skip)


def test_native_pipeline_disable_switch_uses_generic_rgb_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIZDOOM_TURBO_DISABLE_NATIVE_PIPELINE", "1")
    env = make_exact_env(num_envs=1, num_threads=1)
    try:
        env.reset(seed=32)
        assert env._native_stepper is None
        assert env._use_indexed_native is False
        assert env._games[0].get_screen_format() == vzd.ScreenFormat.RGB24
        env.step(np.zeros(1, dtype=np.int64))
    finally:
        env.close()


def test_missing_native_core_uses_generic_rgb_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(vzd, "_TurboBatchStepper")
    env = make_exact_env(num_envs=1, num_threads=1)
    try:
        env.reset(seed=32)
        assert env._native_stepper is None
        assert env._use_indexed_native is False
        assert env._games[0].get_screen_format() == vzd.ScreenFormat.RGB24
        env.step(np.zeros(1, dtype=np.int64))
    finally:
        env.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"maxpool_last_two": True},
        {"obs_resize_algorithm": "bilinear"},
        {"obs_grayscale": False},
        {"obs_resize": (96, 96)},
    ],
)
def test_non_fast_path_profiles_use_generic_implementation(
    overrides: dict[str, object],
) -> None:
    env = make_exact_env(num_envs=1, num_threads=1, **overrides)
    try:
        observations, _infos = env.reset(seed=33)
        assert env._native_stepper is None
        assert env._use_indexed_native is False
        assert env._games[0].get_screen_format() == vzd.ScreenFormat.RGB24
        transition = env.step(np.zeros(1, dtype=np.int64))
        assert transition[0].shape == observations.shape
    finally:
        env.close()


@pytest.mark.parametrize("frame_skip", [1, 4])
def test_native_pipeline_matches_fallback_through_terminals_and_masked_resets(
    monkeypatch: pytest.MonkeyPatch,
    frame_skip: int,
) -> None:
    native = make_exact_env(
        frame_skip=frame_skip,
        vizdoom_config={"episode_timeout": 300, "episode_start_time": 1},
    )
    monkeypatch.setenv("VIZDOOM_TURBO_DISABLE_NATIVE_PIPELINE", "1")
    fallback = make_exact_env(
        frame_skip=frame_skip,
        vizdoom_config={"episode_timeout": 300, "episode_start_time": 1},
    )
    rng = np.random.default_rng(90210)
    resets = 0
    terminations = 0
    truncations = 0
    try:
        assert native.num_envs == 4
        assert (native.raw_height, native.raw_width) == (240, 320)
        assert (native.obs_height, native.obs_width) == (84, 84)
        assert native.obs_grayscale is True
        assert native.obs_layout == "chw"
        assert native.frame_skip == frame_skip
        assert native.maxpool_last_two is False
        assert native._native_stepper is not None
        assert native._use_indexed_native is True
        assert all(
            game.get_screen_format() == vzd.ScreenFormat.DOOM_256_COLORS8
            for game in native._games
        )
        assert fallback._native_stepper is None
        assert fallback._use_indexed_native is False
        assert all(
            game.get_screen_format() == vzd.ScreenFormat.RGB24
            for game in fallback._games
        )
        native_observations, native_infos = native.reset(seed=71)
        fallback_observations, fallback_infos = fallback.reset(seed=71)
        np.testing.assert_array_equal(native_observations, fallback_observations)
        assert_info_equal(native_infos, fallback_infos)

        for step in range(640 // frame_skip):
            actions = rng.integers(
                native.single_action_space.n,
                size=native.num_envs,
                dtype=np.int64,
            )
            native_transition = native.step(actions)
            fallback_transition = fallback.step(actions)
            for native_value, fallback_value in zip(
                native_transition[:4],
                fallback_transition[:4],
                strict=True,
            ):
                np.testing.assert_array_equal(native_value, fallback_value)
            assert_info_equal(native_transition[4], fallback_transition[4])

            done = native_transition[2] | native_transition[3]
            terminations += int(native_transition[2].sum())
            truncations += int(native_transition[3].sum())
            if np.any(done):
                state_indices = np.zeros(native.num_envs, dtype=np.int32)
                seeds = [
                    10_000 + step * native.num_envs + lane if masked else None
                    for lane, masked in enumerate(done)
                ]
                options = {
                    "reset_mask": done,
                    "state_indices": state_indices,
                }
                native_observations, native_infos = native.reset(
                    seed=seeds,
                    options=options,
                )
                fallback_observations, fallback_infos = fallback.reset(
                    seed=seeds,
                    options=options,
                )
                np.testing.assert_array_equal(
                    native_observations,
                    fallback_observations,
                )
                assert_info_equal(native_infos, fallback_infos)
                resets += 1

        assert terminations > 0
        assert truncations > 0
        assert resets >= 2
    finally:
        native.close()
        fallback.close()
