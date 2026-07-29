from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

_ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "surface_variants"
_CATALOG_PATH = _ASSET_ROOT / "defend_the_line" / "catalog.json"
_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9-]*$")
_SAFE_ROLE = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_CVAR = re.compile(r"^[a-z_][a-z0-9_]*$")
_SAFE_TEXTURE = re.compile(r"^[A-Z0-9]{1,8}$")


@dataclass(frozen=True)
class SurfaceVariant:
    role: str
    selector_cvar: str
    variant_id: str
    scenario_index: int
    texture: str
    namespace: str
    theme: str | None
    asset: Path | None
    asset_sha256: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_document() -> dict[str, Any]:
    try:
        document = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid surface-variant catalog: {_CATALOG_PATH}") from exc
    if document.get("schema_version") != 1:
        raise RuntimeError("surface-variant catalog must use schema_version 1")
    return document


def _catalog_asset(relative_path: str, label: str) -> Path:
    path = (_CATALOG_PATH.parent / relative_path).resolve()
    root = _CATALOG_PATH.parent.resolve()
    if root not in path.parents:
        raise RuntimeError(f"{label} escapes the surface-variant asset directory")
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    return path


def _manifest_asset(
    raw: Mapping[str, Any],
    *,
    role: str,
    variant_id: str,
    namespace: str,
    texture: str,
    theme: str | None,
) -> tuple[Path | None, str | None]:
    raw_manifest = raw.get("manifest")
    if variant_id == "original":
        if raw_manifest is not None:
            raise RuntimeError("original surface variants must not declare a manifest")
        return None, None
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise RuntimeError(f"surface variant {variant_id!r} must declare a manifest")
    manifest_path = _catalog_asset(raw_manifest, f"{variant_id} manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid surface-variant manifest: {manifest_path}") from exc
    if (
        manifest.get("id") != variant_id
        or manifest.get("role") != role
        or manifest.get("namespace") != namespace
        or manifest.get("lump") != texture
        or manifest.get("theme") != theme
    ):
        raise RuntimeError(f"surface-variant manifest identity mismatch: {manifest_path}")
    raw_texture = manifest.get("texture")
    if not isinstance(raw_texture, dict):
        raise RuntimeError(f"surface variant {variant_id!r} has no texture metadata")
    relative_asset = str(raw_texture.get("png") or "")
    expected_hash = str(raw_texture.get("png_sha256") or "")
    asset = (manifest_path.parent / relative_asset).resolve()
    if manifest_path.parent not in asset.parents:
        raise RuntimeError("surface-variant texture escapes its manifest directory")
    if not asset.is_file() or _sha256(asset) != expected_hash:
        raise RuntimeError(f"surface-variant texture failed integrity check: {asset}")
    if (
        raw_texture.get("size") != [64, 64]
        or raw_texture.get("fully_opaque") is not True
        or raw_texture.get("colors_in_playpal") is not True
        or raw_texture.get("seamless_left_right") is not True
        or raw_texture.get("seamless_top_bottom") is not True
    ):
        raise RuntimeError(f"surface variant {variant_id!r} failed compatibility checks")
    return asset, expected_hash


def load_defend_line_surface_catalog() -> tuple[Mapping[str, tuple[SurfaceVariant, ...]], str]:
    document = _catalog_document()
    raw_roles = document.get("roles")
    if not isinstance(raw_roles, dict) or not raw_roles:
        raise RuntimeError("surface-variant catalog must contain roles")
    resolved_roles: dict[str, tuple[SurfaceVariant, ...]] = {}
    for role, raw_role in raw_roles.items():
        if not isinstance(role, str) or not _SAFE_ROLE.fullmatch(role):
            raise RuntimeError(f"invalid surface role: {role!r}")
        if not isinstance(raw_role, dict):
            raise RuntimeError(f"surface role {role!r} must be an object")
        selector_cvar = str(raw_role.get("selector_cvar") or "")
        if not _SAFE_CVAR.fullmatch(selector_cvar):
            raise RuntimeError(f"invalid selector cvar for surface role {role!r}")
        raw_variants = raw_role.get("variants")
        if not isinstance(raw_variants, list) or not raw_variants:
            raise RuntimeError(f"surface role {role!r} must contain variants")
        variants: list[SurfaceVariant] = []
        seen_ids: set[str] = set()
        seen_indices: set[int] = set()
        for raw in raw_variants:
            if not isinstance(raw, dict):
                raise RuntimeError("surface-variant catalog entries must be objects")
            variant_id = str(raw.get("id") or "")
            scenario_index = raw.get("scenario_index")
            texture = str(raw.get("texture") or "")
            namespace = str(raw.get("namespace") or "")
            raw_theme = raw.get("theme")
            theme = None if raw_theme is None else str(raw_theme)
            if not _SAFE_IDENTIFIER.fullmatch(variant_id):
                raise RuntimeError(f"invalid surface variant id: {variant_id!r}")
            if (
                isinstance(scenario_index, bool)
                or not isinstance(scenario_index, int)
                or scenario_index < 0
            ):
                raise RuntimeError(f"invalid scenario index for {variant_id!r}")
            if not _SAFE_TEXTURE.fullmatch(texture):
                raise RuntimeError(f"invalid texture name for {variant_id!r}")
            if namespace not in {"texture", "flat"}:
                raise RuntimeError(f"invalid namespace for {variant_id!r}")
            if theme is not None and not _SAFE_IDENTIFIER.fullmatch(theme):
                raise RuntimeError(f"invalid theme for {variant_id!r}")
            if variant_id in seen_ids or scenario_index in seen_indices:
                raise RuntimeError(f"surface role {role!r} has duplicate ids or scenario indices")
            seen_ids.add(variant_id)
            seen_indices.add(scenario_index)
            asset, asset_sha256 = _manifest_asset(
                raw,
                role=role,
                variant_id=variant_id,
                namespace=namespace,
                texture=texture,
                theme=theme,
            )
            variants.append(
                SurfaceVariant(
                    role=role,
                    selector_cvar=selector_cvar,
                    variant_id=variant_id,
                    scenario_index=scenario_index,
                    texture=texture,
                    namespace=namespace,
                    theme=theme,
                    asset=asset,
                    asset_sha256=asset_sha256,
                )
            )
        defaults = raw_role.get("default_variants")
        if (
            not isinstance(defaults, list)
            or not defaults
            or any(value not in seen_ids for value in defaults)
        ):
            raise RuntimeError(f"surface role {role!r} has invalid default variants")
        resolved_roles[role] = tuple(variants)
    _validated_themes(document, resolved_roles)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return MappingProxyType(resolved_roles), hashlib.sha256(canonical).hexdigest()


def _validated_themes(
    document: Mapping[str, Any],
    roles: Mapping[str, tuple[SurfaceVariant, ...]],
) -> Mapping[str, Mapping[str, str]]:
    raw_themes = document.get("themes")
    if not isinstance(raw_themes, dict) or not raw_themes:
        raise RuntimeError("surface-variant catalog must contain themes")
    ids_by_role = {
        role: {variant.variant_id: variant for variant in variants}
        for role, variants in roles.items()
    }
    themes: dict[str, Mapping[str, str]] = {}
    assigned: set[str] = set()
    for theme_id, raw_theme in raw_themes.items():
        if not isinstance(theme_id, str) or not _SAFE_IDENTIFIER.fullmatch(theme_id):
            raise RuntimeError(f"invalid surface theme: {theme_id!r}")
        if not isinstance(raw_theme, dict) or not str(raw_theme.get("display_name") or ""):
            raise RuntimeError(f"surface theme {theme_id!r} must have a display name")
        raw_variants = raw_theme.get("variants")
        if not isinstance(raw_variants, dict) or set(raw_variants) != set(roles):
            raise RuntimeError(f"surface theme {theme_id!r} must cover every role")
        resolved: dict[str, str] = {}
        for role, raw_variant_id in raw_variants.items():
            variant_id = str(raw_variant_id)
            variant = ids_by_role[role].get(variant_id)
            if variant is None or variant.theme != theme_id:
                raise RuntimeError(f"surface theme {theme_id!r} has invalid {role} variant")
            if variant_id in assigned:
                raise RuntimeError(f"surface variant {variant_id!r} belongs to two themes")
            assigned.add(variant_id)
            resolved[role] = variant_id
        themes[theme_id] = MappingProxyType(resolved)
    return MappingProxyType(themes)


def load_defend_line_surface_themes() -> Mapping[str, Mapping[str, str]]:
    catalog, _catalog_hash = load_defend_line_surface_catalog()
    return _validated_themes(_catalog_document(), catalog)


def _requested_ids(
    role: str,
    value: Sequence[str] | None,
    defaults: Sequence[str],
) -> tuple[str, ...]:
    if value is None:
        raw_ids = tuple(str(item) for item in defaults)
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise TypeError(f"surface_variants[{role!r}] must be a sequence of ids")
        raw_ids = tuple(str(item).strip() for item in value)
    if not raw_ids:
        raise ValueError(f"surface_variants[{role!r}] must select at least one variant")
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError(f"surface_variants[{role!r}] cannot contain duplicates")
    return raw_ids


def resolve_defend_line_surface_variants(
    requested: Mapping[str, Sequence[str]] | None,
) -> tuple[Mapping[str, tuple[SurfaceVariant, ...]], str]:
    catalog, catalog_hash = load_defend_line_surface_catalog()
    document_roles = _catalog_document()["roles"]
    if requested is None:
        requested_by_role: Mapping[str, Sequence[str]] = {}
    elif isinstance(requested, Mapping):
        unknown_roles = sorted(set(requested) - set(catalog))
        if unknown_roles:
            raise ValueError(f"unknown Defend the Line surface role(s): {unknown_roles}")
        requested_by_role = requested
    else:
        raise TypeError("surface_variants must be a role mapping")

    selected: dict[str, tuple[SurfaceVariant, ...]] = {}
    for role, variants in catalog.items():
        by_id = {variant.variant_id: variant for variant in variants}
        raw_ids = _requested_ids(
            role,
            requested_by_role.get(role),
            document_roles[role]["default_variants"],
        )
        unknown = [variant_id for variant_id in raw_ids if variant_id not in by_id]
        if unknown:
            choices = ", ".join(by_id)
            raise ValueError(f"unknown {role} surface variant(s): {unknown}; choose from {choices}")
        selected[role] = tuple(by_id[variant_id] for variant_id in raw_ids)
    return MappingProxyType(selected), catalog_hash
