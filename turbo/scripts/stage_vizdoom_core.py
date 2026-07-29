#!/usr/bin/env python3
"""Build and stage the custom ViZDoom package for a Turbo wheel."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
STAGED_PACKAGE = PACKAGE_ROOT / "python" / "vizdoom"
PREBUILT_CORE_ENV = "VIZDOOM_TURBO_PREBUILT_CORE"


def built_package(repository: Path) -> Path:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return repository / "bin" / version / "vizdoom"


def clean() -> None:
    shutil.rmtree(STAGED_PACKAGE, ignore_errors=True)


def copy_core_source(destination: Path) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {
            name
            for name in names
            if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
        }
        if Path(directory) == REPO_ROOT:
            ignored.update(
                {
                    ".git",
                    ".venv",
                    "CMakeCache.txt",
                    "CMakeFiles",
                    "Makefile",
                    "bin",
                    "build",
                    "dist",
                    "turbo",
                    "wheelhouse",
                }
            )
        return ignored

    shutil.copytree(REPO_ROOT, destination, ignore=ignore)


def build_core(repository: Path) -> None:
    environment = os.environ.copy()
    environment.pop("_PYPROJECT_HOOKS_BUILD_BACKEND", None)
    environment.pop("_PYPROJECT_HOOKS_BACKEND_PATH", None)
    build_environment = repository / ".venv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            str(build_environment),
        ],
        cwd=repository,
        env=environment,
        check=True,
    )
    python = (
        build_environment / "Scripts" / "python.exe"
        if os.name == "nt"
        else build_environment / "bin" / "python"
    )
    site_packages = (
        build_environment / "Lib" / "site-packages"
        if os.name == "nt"
        else build_environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    outer_site_packages = [
        path
        for path in sys.path
        if path and Path(path).name in {"site-packages", "dist-packages"}
    ]
    (site_packages / "vizdoom_turbo_build_environment.pth").write_text(
        "".join(f"{path}\n" for path in outer_site_packages),
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--editable",
            str(repository),
            "--no-deps",
            "--no-build-isolation",
        ],
        cwd=repository,
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            (
                "import vizdoom; "
                "assert hasattr(vizdoom, '_TurboBatchStepper'), "
                "'custom _TurboBatchStepper is missing'"
            ),
        ],
        cwd=repository,
        env=environment,
        check=True,
    )


def build_and_stage() -> None:
    clean()
    prebuilt = os.environ.get(PREBUILT_CORE_ENV)
    if prebuilt:
        source = Path(prebuilt).expanduser().resolve()
        required = (source / "__init__.py", source / "vizdoom")
        if not source.is_dir() or any(not path.is_file() for path in required):
            raise RuntimeError(
                f"{PREBUILT_CORE_ENV} must name a built vizdoom package directory"
            )
        if not any(source.glob("vizdoom*.so")) and not any(source.glob("vizdoom*.pyd")):
            raise RuntimeError(
                f"{PREBUILT_CORE_ENV} does not contain a Python extension"
            )
        shutil.copytree(
            source,
            STAGED_PACKAGE,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        return
    with tempfile.TemporaryDirectory(prefix="vizdoom-turbo-core-") as directory:
        repository = Path(directory) / "ViZDoom"
        copy_core_source(repository)
        build_core(repository)
        source = built_package(repository)
        if not source.is_dir():
            raise RuntimeError(f"custom ViZDoom package was not built: {source}")
        shutil.copytree(
            source,
            STAGED_PACKAGE,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


@contextlib.contextmanager
def staged_vizdoom_core() -> Iterator[None]:
    build_and_stage()
    try:
        yield
    finally:
        clean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "clean"))
    args = parser.parse_args()
    if args.command == "build":
        build_and_stage()
    else:
        clean()


if __name__ == "__main__":
    main()
