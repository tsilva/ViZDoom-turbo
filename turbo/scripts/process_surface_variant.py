#!/usr/bin/env python3
"""Normalize an image-generated surface into a seamless PLAYPAL texture."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def seamless_tile(source: Image.Image) -> Image.Image:
    rgb = source.convert("RGB")
    edge = min(rgb.size)
    left = (rgb.width - edge) // 2
    top = (rgb.height - edge) // 2
    base = rgb.crop((left, top, left + edge, top + edge)).resize((32, 32), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (64, 64))
    tile.paste(base, (0, 0))
    tile.paste(ImageOps.mirror(base), (32, 0))
    tile.paste(ImageOps.flip(base), (0, 32))
    tile.paste(ImageOps.flip(ImageOps.mirror(base)), (32, 32))
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
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
        normalized = seamless_tile(selected_source)
    quantized = normalized.quantize(palette=palette, dither=Image.Dither.NONE)

    texture_dir = args.output_dir / "texture"
    proof_dir = args.output_dir / "proof"
    texture_dir.mkdir(parents=True, exist_ok=True)
    proof_dir.mkdir(parents=True, exist_ok=True)
    texture_path = texture_dir / f"{args.lump}.png"
    preview_path = proof_dir / "tiled-preview.png"
    quantized.save(texture_path)

    tile_rgb = quantized.convert("RGB")
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
    manifest = {
        "schema_version": 1,
        "id": args.id,
        "display_name": args.display_name,
        "role": args.role,
        "namespace": args.namespace,
        "lump": args.lump,
        "source": {
            "generation_mode": "built-in imagegen",
            "generated_image_sha256": sha256(args.source),
            "prompt_sha256": sha256(args.prompt),
            "reference_assets": [],
            "freedoom_wad_sha256": sha256(args.freedoom_wad),
        },
        "texture": {
            "png": f"texture/{args.lump}.png",
            "png_sha256": sha256(texture_path),
            "size": [64, 64],
            "fully_opaque": True,
            "colors_in_playpal": pixels <= playpal_colors,
            "seamless_left_right": left_edge == right_edge,
            "seamless_top_bottom": top_edge == bottom_edge,
        },
        "proof": {
            "tiled_preview": "proof/tiled-preview.png",
            "tiled_preview_sha256": sha256(preview_path),
        },
        "provenance": {
            "origin": "Original AI-generated material texture",
            "postprocess": (
                "Center crop; Lanczos downsample to 32x32; mirrored 2x2 wrap; "
                "nearest PLAYPAL quantization without dithering"
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
            "Select gutter-inset source grid cell; center crop; Lanczos downsample "
            "to 32x32; mirrored 2x2 wrap; nearest PLAYPAL quantization without dithering"
        )
    if not all(
        (
            manifest["texture"]["colors_in_playpal"],
            manifest["texture"]["seamless_left_right"],
            manifest["texture"]["seamless_top_bottom"],
        )
    ):
        raise RuntimeError("processed texture failed its compatibility checks")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
