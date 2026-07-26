#!/usr/bin/env python3
"""Build and audit vizdoom-turbo release distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = REPO_ROOT / "turbo"
PACKAGE_NAME = "vizdoom-turbo"
IMPORT_NAME = "vizdoom_turbo"
EXTENSION_NAME = "_vizdoom_turbo"
RELEASE_PLATFORMS = (
    "macos-arm64",
    "macos-x86_64",
    "linux-x86_64",
    "linux-aarch64",
)
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def run(args: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=PACKAGE_ROOT, env=env, check=True)


def read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def project_version() -> str:
    project = read_toml(PACKAGE_ROOT / "pyproject.toml")["project"]
    assert isinstance(project, dict)
    version = project["version"]
    assert isinstance(version, str)
    return version


def cargo_version() -> str:
    package = read_toml(PACKAGE_ROOT / "Cargo.toml")["package"]
    assert isinstance(package, dict)
    version = package["version"]
    assert isinstance(version, str)
    return version


def cargo_lock_version() -> str:
    lock = read_toml(PACKAGE_ROOT / "Cargo.lock")
    packages = lock.get("package", [])
    assert isinstance(packages, list)
    for package in packages:
        if isinstance(package, dict) and package.get("name") == PACKAGE_NAME:
            version = package.get("version")
            assert isinstance(version, str)
            return version
    raise SystemExit(f"{PACKAGE_NAME!r} is missing from Cargo.lock")


def validate_version(version: str) -> None:
    if not VERSION_PATTERN.fullmatch(version):
        raise SystemExit(f"release version must be MAJOR.MINOR.PATCH: {version!r}")


def check_version(args: argparse.Namespace) -> None:
    expected = args.version or project_version()
    validate_version(expected)
    project = read_toml(PACKAGE_ROOT / "pyproject.toml")["project"]
    assert isinstance(project, dict)
    actual = {
        "project.name": project.get("name"),
        "pyproject.toml": project_version(),
        "Cargo.toml": cargo_version(),
        "Cargo.lock": cargo_lock_version(),
    }
    failures = {
        key: value
        for key, value in actual.items()
        if value != (PACKAGE_NAME if key == "project.name" else expected)
    }
    if failures:
        raise SystemExit(
            f"release metadata mismatch for {expected}: "
            + ", ".join(f"{key}={value!r}" for key, value in failures.items())
        )
    print(json.dumps({"package": PACKAGE_NAME, "version": expected}, indent=2))


def split_version(version: str) -> tuple[int, int, int]:
    validate_version(version)
    return tuple(int(part) for part in version.split("."))  # type: ignore[return-value]


def next_version(version: str, part: str) -> str:
    major, minor, patch = split_version(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def replace_section_version(path: Path, section: str, version: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf'(?ms)(^\[{re.escape(section)}\]\n.*?^version\s*=\s*")[^"]+(")',
    )
    updated, count = pattern.subn(rf"\g<1>{version}\g<2>", text, count=1)
    if count != 1:
        raise SystemExit(f"could not update [{section}] version in {path.name}")
    path.write_text(updated, encoding="utf-8")


def bump_version(args: argparse.Namespace) -> None:
    current = project_version()
    target = args.to or next_version(current, args.part)
    validate_version(target)
    if args.write:
        replace_section_version(PACKAGE_ROOT / "pyproject.toml", "project", target)
        replace_section_version(PACKAGE_ROOT / "Cargo.toml", "package", target)
    print(target)


def fetch_pypi() -> dict[str, object]:
    url = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {}
        raise
    if not isinstance(data, dict):
        raise SystemExit("unexpected PyPI JSON response")
    return data


def check_pypi(args: argparse.Namespace) -> None:
    validate_version(args.version)
    releases = fetch_pypi().get("releases", {})
    if not isinstance(releases, dict):
        raise SystemExit("unexpected PyPI releases payload")
    if releases.get(args.version):
        raise SystemExit(f"{PACKAGE_NAME}=={args.version} already exists on PyPI")
    print(f"{PACKAGE_NAME}=={args.version} is unused on PyPI")


def wheelhouse(version: str, platform: str) -> Path:
    return PACKAGE_ROOT / f"wheelhouse-v{version}-{platform}"


def build_platform(args: argparse.Namespace) -> None:
    version = args.version or project_version()
    validate_version(version)
    output = wheelhouse(version, args.platform)
    if output.exists():
        raise SystemExit(f"release output already exists: {output}")
    output.mkdir()
    env = os.environ.copy()
    target = PACKAGE_ROOT / f"target-release-{args.platform}"
    env["CARGO_TARGET_DIR"] = str(target)
    if args.platform.startswith("macos-"):
        arch = args.platform.removeprefix("macos-")
        env["ARCHFLAGS"] = f"-arch {arch}"
        env["MACOSX_DEPLOYMENT_TARGET"] = "11.0" if arch == "arm64" else "10.15"
        run(
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--release",
                "--locked",
                "--out",
                str(output),
            ],
            env=env,
        )
        return
    arch = args.platform.removeprefix("linux-")
    env.update(
        {
            "CIBW_ARCHS_LINUX": arch,
            "CIBW_BEFORE_ALL_LINUX": (
                "curl https://sh.rustup.rs -sSf | "
                "sh -s -- -y --profile minimal --default-toolchain stable"
            ),
            "CIBW_BUILD": f"cp311-manylinux_{arch}",
            "CIBW_BUILD_VERBOSITY": "1",
            "CIBW_ENVIRONMENT_LINUX": (
                'PATH="$HOME/.cargo/bin:$PATH" '
                "CARGO_NET_GIT_FETCH_WITH_CLI=true "
                "CARGO_TARGET_DIR=/tmp/cargo-target"
            ),
            "CIBW_SKIP": "*-musllinux_*",
        }
    )
    run(
        [
            sys.executable,
            "-m",
            "cibuildwheel",
            "--platform",
            "linux",
            "--output-dir",
            str(output),
        ],
        env=env,
    )


def resolve_wheel(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    wheels = sorted(path.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel in {path}, found {len(wheels)}")
    return wheels[0]


def wheel_platform(wheel: Path) -> str | None:
    markers = {
        "macos-arm64": ("macosx", "arm64"),
        "macos-x86_64": ("macosx", "x86_64"),
        "linux-x86_64": ("manylinux", "x86_64"),
        "linux-aarch64": ("manylinux", "aarch64"),
    }
    for platform, required in markers.items():
        if all(marker in wheel.name for marker in required):
            return platform
    return None


def audit_wheel(wheel: Path, version: str) -> dict[str, object]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    extension = [
        name
        for name in names
        if name.startswith(f"{IMPORT_NAME}/{EXTENSION_NAME}")
        and name.endswith((".so", ".pyd"))
    ]
    checks = {
        "version_in_filename": version in wheel.name,
        "stable_abi": "abi3" in wheel.name,
        "known_platform": wheel_platform(wheel) is not None,
        "has_init": f"{IMPORT_NAME}/__init__.py" in names,
        "has_environment": f"{IMPORT_NAME}/env.py" in names,
        "has_action_tables": f"{IMPORT_NAME}/action_tables.py" in names,
        "has_py_typed": f"{IMPORT_NAME}/py.typed" in names,
        "has_extension": len(extension) == 1,
        "has_metadata": sum(name.endswith(".dist-info/METADATA") for name in names) == 1,
        "has_license": any(name.endswith(".dist-info/licenses/LICENSE") for name in names),
        "no_cache_files": not any(
            "__pycache__" in Path(name).parts or name.endswith(".pyc") for name in names
        ),
    }
    return {
        "wheel": str(wheel),
        "platform": wheel_platform(wheel),
        "extension": extension,
        "checks": checks,
    }


def assert_audits(results: list[dict[str, object]]) -> None:
    failures = {}
    for result in results:
        checks = result["checks"]
        assert isinstance(checks, dict)
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            failures[str(result["wheel"])] = failed
    if failures:
        print(json.dumps(results, indent=2), file=sys.stderr)
        raise SystemExit(f"wheel audit failed: {failures}")


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def smoke_wheel(args: argparse.Namespace) -> None:
    wheel = resolve_wheel(args.wheel)
    version = args.version or project_version()
    result = audit_wheel(wheel, version)
    assert_audits([result])
    with tempfile.TemporaryDirectory(prefix="vizdoom-turbo-smoke-") as directory:
        environment = Path(directory) / "venv"
        run(["uv", "venv", "--python", sys.executable, str(environment)])
        python = venv_python(environment)
        run(["uv", "pip", "install", "--python", str(python), str(wheel)])
        code = """
import numpy as np
from importlib.metadata import version
from vizdoom_turbo import VizdoomTurboVecEnv, scenario_buttons

assert version("vizdoom-turbo") == %r
assert scenario_buttons("VizdoomBasic-v1") == ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK")
env = VizdoomTurboVecEnv(
    "VizdoomBasic-v1",
    num_envs=2,
    num_threads=2,
    obs_resize=(32, 40),
    frame_skip=2,
    frame_stack=4,
    use_restricted_actions="minimal",
)
try:
    observations, _ = env.reset(seed=7)
    assert observations.shape == (2, 4, 32, 40)
    env.step(np.zeros(2, dtype=np.int64))
finally:
    env.close()
""" % version
        run([str(python), "-c", code])
    print(json.dumps(result, indent=2))


def audit_sdist(args: argparse.Namespace) -> None:
    version = args.version or project_version()
    sdist = args.sdist.resolve()
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
    checks = {
        "version_in_filename": version in sdist.name,
        "has_pyproject": any(name.endswith("/pyproject.toml") for name in names),
        "has_cargo_toml": any(name.endswith("/Cargo.toml") for name in names),
        "has_cargo_lock": any(name.endswith("/Cargo.lock") for name in names),
        "has_license": any(name.endswith("/LICENSE") for name in names),
        "has_readme": any(name.endswith("/README.md") for name in names),
        "has_python_package": any(
            name.endswith(f"/{IMPORT_NAME}/env.py") for name in names
        ),
        "has_rust_source": any(name.endswith("/src/lib.rs") for name in names),
        "no_build_outputs": not any(
            part in {"target", "dist", ".venv", ".git"}
            for name in names
            for part in Path(name).parts
        ),
    }
    result = {"sdist": str(sdist), "checks": checks}
    print(json.dumps(result, indent=2))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"sdist audit failed: {failed}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_check(args: argparse.Namespace) -> None:
    version = args.version or project_version()
    wheels = [Path(value).resolve() for value in args.wheels]
    if not wheels:
        raise SystemExit("final-check requires the complete wheel set")
    results = [audit_wheel(wheel, version) for wheel in wheels]
    assert_audits(results)
    seen = {result["platform"] for result in results}
    missing = sorted(set(RELEASE_PLATFORMS) - seen)
    if missing:
        raise SystemExit(f"release wheel set is missing: {', '.join(missing)}")
    if len(wheels) != len(RELEASE_PLATFORMS):
        raise SystemExit(f"expected {len(RELEASE_PLATFORMS)} wheels, found {len(wheels)}")
    run([sys.executable, "-m", "twine", "check", *[str(wheel) for wheel in wheels]])
    print(
        json.dumps(
            {
                "audits": results,
                "sha256": {wheel.name: sha256(wheel) for wheel in wheels},
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-version")
    check.add_argument("--version")
    check.set_defaults(func=check_version)

    bump = commands.add_parser("bump-version")
    bump.add_argument("--to")
    bump.add_argument("--part", choices=("major", "minor", "patch"), default="patch")
    bump.add_argument("--write", action="store_true")
    bump.set_defaults(func=bump_version)

    pypi = commands.add_parser("check-pypi")
    pypi.add_argument("--version", required=True)
    pypi.set_defaults(func=check_pypi)

    build = commands.add_parser("build-platform")
    build.add_argument("--platform", choices=RELEASE_PLATFORMS, required=True)
    build.add_argument("--version")
    build.set_defaults(func=build_platform)

    smoke = commands.add_parser("smoke-wheel")
    smoke.add_argument("wheel", type=Path)
    smoke.add_argument("--version")
    smoke.set_defaults(func=smoke_wheel)

    sdist = commands.add_parser("audit-sdist")
    sdist.add_argument("sdist", type=Path)
    sdist.add_argument("--version")
    sdist.set_defaults(func=audit_sdist)

    final = commands.add_parser("final-check")
    final.add_argument("wheels", nargs="*")
    final.add_argument("--version")
    final.set_defaults(func=final_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
