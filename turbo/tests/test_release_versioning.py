from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def release_script() -> ModuleType:
    return load_module("vizdoom_turbo_release", REPO_ROOT / "turbo/scripts/release.py")


@pytest.fixture(scope="module")
def release_build() -> ModuleType:
    return load_module(
        "vizdoom_turbo_release_build",
        REPO_ROOT / ".codex/skills/build-release/scripts/release_build.py",
    )


@pytest.mark.parametrize(
    ("current", "upstream", "expected"),
    [
        ("0.1.3", "1.3.0", "1.3.0.post1"),
        ("1.3.0", "1.3.0", "1.3.0.post1"),
        ("1.3.0.post1", "1.3.0", "1.3.0.post2"),
        ("1.3.0.post35", "1.4.0", "1.4.0.post1"),
    ],
)
def test_next_post_version_matches_upstream(
    release_script: ModuleType,
    current: str,
    upstream: str,
    expected: str,
) -> None:
    assert release_script.next_post_version(current, upstream) == expected


def test_pinned_upstream_version_is_release_base(
    release_script: ModuleType,
    release_build: ModuleType,
) -> None:
    assert release_script.upstream_vizdoom_version() == "1.3.0"
    assert release_build.upstream_vizdoom_version() == "1.3.0"
    assert release_build.parse_version(release_build.project_version())[0] == "1.3.0"
    assert release_build.cargo_version() == "1.3.0"


def test_custom_core_is_bundled_instead_of_a_runtime_dependency(
    release_build: ModuleType,
) -> None:
    metadata = release_build.read_toml(REPO_ROOT / "turbo/pyproject.toml")
    project = metadata["project"]
    assert isinstance(project, dict)
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    assert not any(
        isinstance(dependency, str) and dependency.startswith("vizdoom")
        for dependency in dependencies
    )
    assert metadata["build-system"]["build-backend"] == "build_backend"


def test_release_matrix_covers_each_supported_cpython(
    release_build: ModuleType,
) -> None:
    assert release_build.PYTHON_TAGS == ("cp311", "cp312", "cp313", "cp314")
