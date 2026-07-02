from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from buffalo_weight.config import load_config
from buffalo_weight.feature_calculators import calculate_mask_features
from buffalo_weight.index import read_index
from buffalo_weight.validation import validate_mask_files

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def available_mask_stems(masks_dir: Path) -> set[str]:
    return {
        path.stem for suffix in IMAGE_SUFFIXES for path in masks_dir.glob(f"*{suffix}")
    }


def find_mask_path(masks_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        path = masks_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def format_value(value: object) -> object:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return value


def generate_feature_index(config_path: Path) -> None:
    config = load_config(config_path)
    data = config["data"]
    features = config["features"]
    if not isinstance(data, dict):
        raise ValueError("config data section must be a map")
    if not isinstance(features, dict):
        raise ValueError("config features section must be a map")
    validate_mask_files(config)

    index_path = Path(str(data["index_path"]))
    masks_dir = Path(str(data["masks_dir"]))
    name_column = str(data["file_name_column"])
    farm_column = str(data["farm_column"])
    target_column = str(data["target_column"])
    tag_column = str(data["tag_column"])
    output_path = Path(str(features["features_index_path"]))
    columns = features["columns"]
    if not isinstance(columns, list):
        raise ValueError("config features.columns must be a list")

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
            feature_values = calculate_mask_features(find_mask_path(masks_dir, stem))
            output_row = {
                "file_name": stem,
                "farm": row.get(farm_column, ""),
                "weight": row.get(target_column, ""),
                "tag": row.get(tag_column, ""),
                **feature_values,
            }
            writer.writerow(
                {
                    column: format_value(output_row.get(str(column), ""))
                    for column in columns
                }
            )
    temp_path.replace(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", required=True)
    args = parser.parse_args(argv)

    try:
        generate_feature_index(Path(args.shared_config))
    except (SystemExit, KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
