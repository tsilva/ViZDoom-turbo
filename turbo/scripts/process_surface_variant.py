#!/usr/bin/env python3
"""Normalize an image-generated surface into a seamless PLAYPAL texture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from PIL import Image, ImageOps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wad_lump(path: Path, requested_name: str) -> bytes:
    data = path.read_bytes()
    magic, count, directory = struct.unpack_from("<4sii", data, 0)
    if magic not in {b"IWAD", b"PWAD"}:
        raise ValueError(f"{path} is not a WAD")
    for index in range(count):
        offset, size, raw_name = struct.unpack_from("<ii8s", data, directory + index * 16)
        name = raw_name.rstrip(b"\0").decode("ascii")
        if name == requested_name:
            return data[offset : offset + size]
    raise ValueError(f"{path} does not contain {requested_name}")


def playpal_image(wad_path: Path) -> tuple[Image.Image, set[tuple[int, int, int]]]:
    palette_bytes = wad_lump(wad_path, "PLAYPAL")[: 256 * 3]
    if len(palette_bytes) != 256 * 3:
        raise ValueError("PLAYPAL does not contain a complete base palette")
    palette = Image.new("P", (1, 1))
    palette.putpalette(palette_bytes)
    colors = {tuple(palette_bytes[index : index + 3]) for index in range(0, len(palette_bytes), 3)}
    return palette, colors


def grid_cell(
    source: Image.Image,
    *,
    row: int | None,
    column: int | None,
    rows: int,
    columns: int,
) -> Image.Image:
    rgb = ImageOps.exif_transpose(source).convert("RGB")
    if row is None and column is None:
        return rgb
    if row is None or column is None:
        raise ValueError("--grid-row and --grid-column must be supplied together")
    if not 0 <= row < rows or not 0 <= column < columns:
        raise ValueError("grid cell is outside the declared grid")
    left = round(column * rgb.width / columns)
    right = round((column + 1) * rgb.width / columns)
    top = round(row * rgb.height / rows)
    bottom = round((row + 1) * rgb.height / rows)
    inset = max(1, round(min(right - left, bottom - top) * 0.02))
    return rgb.crop((left + inset, top + inset, right - inset, bottom - inset))


def direct_tile(source: Image.Image) -> Image.Image:
    rgb = source.convert("RGB")
    edge = min(rgb.size)
    left = (rgb.width - edge) // 2
    top = (rgb.height - edge) // 2
    return rgb.crop((left, top, left + edge, top + edge)).resize(
        (64, 64),
        Image.Resampling.LANCZOS,
    )


def color_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def wrap_seam_ratio(image: Image.Image, axis: str) -> float:
    rgb = image.convert("RGB")
    pixels = rgb.load()
    if axis == "x":
        seam = (
            sum(color_distance(pixels[0, y], pixels[rgb.width - 1, y]) for y in range(rgb.height))
            / rgb.height
        )
        adjacent = sum(
            color_distance(pixels[x, y], pixels[x + 1, y])
            for y in range(rgb.height)
            for x in range(rgb.width - 1)
        ) / (rgb.height * (rgb.width - 1))
    elif axis == "y":
        seam = (
            sum(color_distance(pixels[x, 0], pixels[x, rgb.height - 1]) for x in range(rgb.width))
            / rgb.width
        )
        adjacent = sum(
            color_distance(pixels[x, y], pixels[x, y + 1])
            for x in range(rgb.width)
            for y in range(rgb.height - 1)
        ) / (rgb.width * (rgb.height - 1))
    else:
        raise ValueError(f"unsupported wrap axis: {axis}")
    if adjacent == 0:
        return 0.0 if seam == 0 else 1_000_000.0
    return seam / adjacent


def reconcile_wrap(image: Image.Image, axis: str, width: int) -> Image.Image:
    rgb = image.convert("RGB")
    dimension = rgb.width if axis == "x" else rgb.height
    if width < 2 or width * 2 >= dimension:
        raise ValueError("--seam-blend-width must be at least 2 and less than half the tile size")
    source = rgb.load()
    repaired = rgb.copy()
    target = repaired.load()
    for offset in range(width):
        phase = offset / (width - 1)
        strength = 0.5 * (1.0 + math.cos(math.pi * phase))
        span = rgb.height if axis == "x" else rgb.width
        for position in range(span):
            if axis == "x":
                first_coord = (offset, position)
                last_coord = (rgb.width - 1 - offset, position)
            else:
                first_coord = (position, offset)
                last_coord = (position, rgb.height - 1 - offset)
            first = source[first_coord]
            last = source[last_coord]
            midpoint = tuple(round((first[index] + last[index]) / 2) for index in range(3))
            target[first_coord] = tuple(
                round(first[index] * (1.0 - strength) + midpoint[index] * strength)
                for index in range(3)
            )
            target[last_coord] = tuple(
                round(last[index] * (1.0 - strength) + midpoint[index] * strength)
                for index in range(3)
            )
    return repaired


def compile_tile(
    source: Image.Image,
    *,
    palette: Image.Image,
    seam_threshold: float,
    seam_blend_width: int,
) -> tuple[Image.Image, dict[str, object]]:
    normalized = direct_tile(source)
    initial = normalized.quantize(palette=palette, dither=Image.Dither.NONE)
    initial_ratios = {axis: wrap_seam_ratio(initial, axis) for axis in ("x", "y")}
    repaired_axes: list[str] = []
    quantized = initial
    final_ratios = dict(initial_ratios)
    for _ in range(2):
        failing = [
            axis
            for axis in ("x", "y")
            if final_ratios[axis] > seam_threshold and axis not in repaired_axes
        ]
        if not failing:
            break
        for axis in failing:
            normalized = reconcile_wrap(normalized, axis, seam_blend_width)
            repaired_axes.append(axis)
        quantized = normalized.quantize(palette=palette, dither=Image.Dither.NONE)
        final_ratios = {axis: wrap_seam_ratio(quantized, axis) for axis in ("x", "y")}
    return quantized, {
        "threshold": seam_threshold,
        "blend_width": seam_blend_width,
        "initial_ratios": initial_ratios,
        "repaired_axes": repaired_axes,
        "final_ratios": final_ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--reference-asset", type=Path, action="append", default=[])
    parser.add_argument("--freedoom-wad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--theme")
    parser.add_argument("--role", choices=("wall", "floor", "ceiling"), required=True)
    parser.add_argument("--namespace", choices=("texture", "flat"), required=True)
    parser.add_argument("--lump", required=True)
    parser.add_argument("--grid-row", type=int)
    parser.add_argument("--grid-column", type=int)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-columns", type=int, default=3)
    parser.add_argument("--seam-threshold", type=float, default=1.5)
    parser.add_argument("--seam-blend-width", type=int, default=4)
    args = parser.parse_args()

    if (
        len(args.lump) > 8
        or not args.lump.isascii()
        or not args.lump.isalnum()
        or args.lump.upper() != args.lump
    ):
        raise ValueError("--lump must be an uppercase alphanumeric Doom lump name")

    palette, playpal_colors = playpal_image(args.freedoom_wad)
    with Image.open(args.source) as generated:
        selected_source = grid_cell(
            generated,
            row=args.grid_row,
            column=args.grid_column,
            rows=args.grid_rows,
            columns=args.grid_columns,
        )
        quantized, wrap = compile_tile(
            selected_source,
            palette=palette,
            seam_threshold=args.seam_threshold,
            seam_blend_width=args.seam_blend_width,
        )

    texture_dir = args.output_dir / "texture"
    proof_dir = args.output_dir / "proof"
    texture_dir.mkdir(parents=True, exist_ok=True)
    proof_dir.mkdir(parents=True, exist_ok=True)
    packaged_prompt_path = args.output_dir / "PROMPT.md"
    texture_path = texture_dir / f"{args.lump}.png"
    single_preview_path = proof_dir / "single-preview.png"
    preview_path = proof_dir / "tiled-preview.png"
    packaged_prompt_path.write_bytes(args.prompt.read_bytes())
    quantized.save(texture_path)

    tile_rgb = quantized.convert("RGB")
    tile_rgb.resize((512, 512), Image.Resampling.NEAREST).save(single_preview_path)
    preview = Image.new("RGB", (256, 256))
    for y in range(0, preview.height, tile_rgb.height):
        for x in range(0, preview.width, tile_rgb.width):
            preview.paste(tile_rgb, (x, y))
    preview.save(preview_path)

    raw_pixels = tile_rgb.tobytes()
    pixels = {tuple(raw_pixels[index : index + 3]) for index in range(0, len(raw_pixels), 3)}
    left_edge = [tile_rgb.getpixel((0, y)) for y in range(tile_rgb.height)]
    right_edge = [tile_rgb.getpixel((tile_rgb.width - 1, y)) for y in range(tile_rgb.height)]
    top_edge = [tile_rgb.getpixel((x, 0)) for x in range(tile_rgb.width)]
    bottom_edge = [tile_rgb.getpixel((x, tile_rgb.height - 1)) for x in range(tile_rgb.width)]
    final_wrap = wrap["final_ratios"]
    manifest = {
        "schema_version": 2,
        "id": args.id,
        "display_name": args.display_name,
        "role": args.role,
        "namespace": args.namespace,
        "lump": args.lump,
        "source": {
            "generation_mode": "built-in imagegen",
            "generated_image_sha256": sha256(args.source),
            "prompt": packaged_prompt_path.name,
            "prompt_sha256": sha256(packaged_prompt_path),
            "reference_assets": [
                {
                    "path": str(reference),
                    "sha256": sha256(reference),
                }
                for reference in args.reference_asset
            ],
            "freedoom_wad_sha256": sha256(args.freedoom_wad),
        },
        "processing": {
            "resize": {
                "size": [64, 64],
                "filter": "Lanczos",
                "passes": 1,
            },
            "palette": {
                "source": "base Freedoom PLAYPAL",
                "dither": False,
            },
            "wrap": wrap,
        },
        "texture": {
            "png": f"texture/{args.lump}.png",
            "png_sha256": sha256(texture_path),
            "size": [64, 64],
            "fully_opaque": True,
            "colors_in_playpal": pixels <= playpal_colors,
            "wrap_x_within_threshold": final_wrap["x"] <= args.seam_threshold,
            "wrap_y_within_threshold": final_wrap["y"] <= args.seam_threshold,
            "opposite_edges_equal_x": left_edge == right_edge,
            "opposite_edges_equal_y": top_edge == bottom_edge,
        },
        "proof": {
            "single_preview": "proof/single-preview.png",
            "single_preview_sha256": sha256(single_preview_path),
            "tiled_preview": "proof/tiled-preview.png",
            "tiled_preview_sha256": sha256(preview_path),
        },
        "provenance": {
            "origin": "Original AI-generated material texture",
            "postprocess": (
                "Center crop; one-pass Lanczos downsample to 64x64; conditional "
                "narrow raised-cosine wrap reconciliation; nearest PLAYPAL "
                "quantization without dithering"
            ),
        },
    }
    if args.theme:
        manifest["theme"] = args.theme
    if args.grid_row is not None:
        manifest["source"]["grid"] = {
            "rows": args.grid_rows,
            "columns": args.grid_columns,
            "row": args.grid_row,
            "column": args.grid_column,
            "cell_inset_fraction": 0.02,
        }
        manifest["provenance"]["postprocess"] = (
            "Select gutter-inset source grid cell; center crop; one-pass Lanczos "
            "downsample to 64x64; conditional narrow raised-cosine wrap "
            "reconciliation; nearest PLAYPAL quantization without dithering"
        )
    if not all(
        (
            manifest["texture"]["colors_in_playpal"],
            manifest["texture"]["wrap_x_within_threshold"],
            manifest["texture"]["wrap_y_within_threshold"],
        )
    ):
        raise RuntimeError("processed texture failed its compatibility checks")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
