from __future__ import annotations

import argparse
import csv
import struct
import sys
from pathlib import Path
import zlib

from buffalo_weight.config import load_config
from buffalo_weight.feature_calculators import calculate_mask_features, zero_features
from buffalo_weight.feature_calculators.circularity import calculate_circularity
from buffalo_weight.feature_calculators.equivalent_diameter import (
    calculate_equivalent_diameter,
)
from buffalo_weight.feature_calculators.geometry import convex_hull, polygon_area
from buffalo_weight.index import read_index

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def available_mask_stems(masks_dir: Path) -> set[str]:
    return {
        path.stem
        for suffix in IMAGE_SUFFIXES
        for path in masks_dir.glob(f"*{suffix}")
    }


def find_mask_path(masks_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        path = masks_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def read_png_mask(path: Path) -> list[list[bool]]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Unsupported image format: {path}")

    offset = 8
    width = height = bit_depth = color_type = 0
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(
                ">IIBBBBB", payload
            )
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break

    if bit_depth != 8 or color_type not in {0, 2, 6}:
        raise ValueError(f"Unsupported PNG type: {path}")

    channels = {0: 1, 2: 3, 6: 4}[color_type]
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    rows: list[bytes] = []
    position = 0
    previous = bytes(stride)

    for _ in range(height):
        filter_type = raw[position]
        position += 1
        scanline = bytearray(raw[position : position + stride])
        position += stride

        for index, value in enumerate(scanline):
            left = scanline[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                scanline[index] = (value + left) & 0xFF
            elif filter_type == 2:
                scanline[index] = (value + above) & 0xFF
            elif filter_type == 3:
                scanline[index] = (value + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                scanline[index] = (value + paeth_predictor(left, above, upper_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"Unsupported PNG filter: {filter_type}")

        previous = bytes(scanline)
        rows.append(previous)

    mask = []
    for row in rows:
        mask_row = []
        for x in range(width):
            start = x * channels
            mask_row.append(any(channel > 0 for channel in row[start : start + min(channels, 3)]))
        mask.append(mask_row)
    return mask


def calculate_features_fast(path: Path) -> dict[str, float]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return calculate_mask_features(read_png_mask(path))

    mask = np.asarray(Image.open(path).convert("L")) > 0
    area = int(mask.sum())
    if area == 0:
        return zero_features()

    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    perimeter = int(
        ((center & ~up).sum())
        + ((center & ~down).sum())
        + ((center & ~left).sum())
        + ((center & ~right).sum())
    )

    boundary = center & (~up | ~down | ~left | ~right)
    boundary_y, boundary_x = np.nonzero(boundary)
    corners = []
    for x, y in zip(boundary_x.tolist(), boundary_y.tolist(), strict=True):
        corners.extend([(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)])
    hull_area = polygon_area(convex_hull(corners))

    ys, xs = np.nonzero(mask)
    x_values = xs.astype(float) + 0.5
    y_values = ys.astype(float) + 0.5
    m00 = float(area)
    cx = float(x_values.sum() / m00)
    cy = float(y_values.sum() / m00)
    dx = x_values - cx
    dy = y_values - cy
    eta20 = float((dx**2).sum() / (m00**2))
    eta02 = float((dy**2).sum() / (m00**2))
    eta11 = float((dx * dy).sum() / (m00**2))

    return {
        "area": area,
        "perimeter": perimeter,
        "solidity": area / hull_area if hull_area else 0,
        "circularity": calculate_circularity(area, perimeter),
        "equivalent_diameter": calculate_equivalent_diameter(area),
        "hu_moment_1": eta20 + eta02,
        "hu_moment_2": (eta20 - eta02) ** 2 + 4 * eta11**2,
    }


def format_value(value: object) -> object:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return value


def generate_feature_index(config_path: Path) -> None:
    config = load_config(config_path)
    data = config["data"]
    output = config["output"]
    if not isinstance(data, dict):
        raise ValueError("config data section must be a map")
    if not isinstance(output, dict):
        raise ValueError("config output section must be a map")

    index_path = Path(str(data["index_path"]))
    masks_dir = Path(str(data["masks_dir"]))
    name_column = str(data["file_name_column"])
    farm_column = str(data["farm_column"])
    target_column = str(data["target_column"])
    tag_column = str(data["tag_column"])
    output_path = Path(str(output["features_index_path"]))
    columns = output["columns"]
    if not isinstance(columns, list):
        raise ValueError("config output.columns must be a list")

    rows = read_index(index_path)
    expected = [row[name_column] for row in rows if row.get(name_column)]
    available = available_mask_stems(masks_dir)
    missing = sorted(stem for stem in expected if stem not in available)

    if missing:
        lines = ["Missing masks:", *missing]
        raise SystemExit("\n".join(lines))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[str(column) for column in columns])
        writer.writeheader()
        for row in rows:
            stem = row[name_column]
            feature_values = calculate_features_fast(find_mask_path(masks_dir, stem))
            output_row = {
                "file_name": stem,
                "farm": row.get(farm_column, ""),
                "weight": row.get(target_column, ""),
                "tag": row.get(tag_column, ""),
                **feature_values,
            }
            writer.writerow(
                {column: format_value(output_row.get(str(column), "")) for column in columns}
            )
    temp_path.replace(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    try:
        generate_feature_index(Path(args.config))
    except SystemExit as error:
        print(error, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
