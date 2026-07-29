from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .action_tables import ActionTable
from .env import VizDoomTurboVecEnv, VizdoomTurboVecEnv, scenario_buttons

try:
    __version__ = version("vizdoom-turbo")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "ActionTable",
    "VizDoomTurboVecEnv",
    "VizdoomTurboVecEnv",
    "__version__",
    "scenario_buttons",
]

try:
    import gymnasium as gym

    for _name, _game in {
        "VizdoomBasic-Turbo-v0": "VizdoomBasic-v1",
        "VizdoomDeadlyCorridor-Turbo-v0": "VizdoomDeadlyCorridor-v1",
        "VizdoomDefendCenter-Turbo-v0": "VizdoomDefendCenter-v1",
        "VizdoomDefendLine-Turbo-v0": "VizdoomDefendLine-v1",
        "VizdoomDefendLine-Plus-v1": "VizdoomDefendLine-Plus-v1",
        "VizdoomHealthGathering-Turbo-v0": "VizdoomHealthGathering-v1",
        "VizdoomHealthGatheringSupreme-Turbo-v0": "VizdoomHealthGatheringSupreme-v1",
        "VizdoomMyWayHome-Turbo-v0": "VizdoomMyWayHome-v1",
        "VizdoomPredictPosition-Turbo-v0": "VizdoomPredictPosition-v1",
        "VizdoomTakeCover-Turbo-v0": "VizdoomTakeCover-v1",
    }.items():
        if _name not in gym.envs.registry:
            gym.register(
                id=_name,
                entry_point=None,
                vector_entry_point="vizdoom_turbo:VizdoomTurboVecEnv",
                kwargs={"game": _game},
            )
except Exception:
    pass
