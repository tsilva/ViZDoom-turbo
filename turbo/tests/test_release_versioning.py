from __future__ import annotations

import importlib.util
import sys
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


@pytest.fixture(scope="module")
def build_backend() -> ModuleType:
    package_root = str(REPO_ROOT / "turbo")
    sys.path.insert(0, package_root)
    try:
        return load_module(
            "vizdoom_turbo_build_backend",
            REPO_ROOT / "turbo/build_backend.py",
        )
    finally:
        sys.path.remove(package_root)


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


def test_editable_build_keeps_staged_custom_core(
    build_backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        build_backend,
        "build_and_stage",
        lambda: events.append("stage"),
    )
    monkeypatch.setattr(build_backend, "clean", lambda: events.append("clean"))

    def build_editable(*args: object) -> str:
        events.append("build")
        return "vizdoom_turbo-editable.whl"

    monkeypatch.setattr(build_backend.maturin, "build_editable", build_editable)

    assert (
        build_backend.build_editable(str(tmp_path))
        == "vizdoom_turbo-editable.whl"
    )
    assert events == ["stage", "build"]


def test_failed_editable_build_cleans_staged_custom_core(
    build_backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        build_backend,
        "build_and_stage",
        lambda: events.append("stage"),
    )
    monkeypatch.setattr(build_backend, "clean", lambda: events.append("clean"))

    def fail_build(*args: object) -> str:
        events.append("build")
        raise RuntimeError("editable build failed")

    monkeypatch.setattr(build_backend.maturin, "build_editable", fail_build)

    with pytest.raises(RuntimeError, match="editable build failed"):
        build_backend.build_editable(str(tmp_path))
    assert events == ["stage", "build", "clean"]


def test_release_matrix_covers_each_supported_cpython(
    release_build: ModuleType,
) -> None:
    assert release_build.PYTHON_TAGS == ("cp311", "cp312", "cp313", "cp314")
