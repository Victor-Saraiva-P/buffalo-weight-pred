from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from buffalo_weight.index import read_index


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def file_names_from_index(shared_config: dict[object, object]) -> set[str]:
    data = shared_config["data"]
    if not isinstance(data, dict):
        raise ValueError("shared config data section must be a map")
    rows = read_index(Path(str(data["index_path"])))
    file_name_column = str(data["file_name_column"])
    return {row[file_name_column] for row in rows if row.get(file_name_column)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def validate_columns(rows: list[dict[str, str]], columns: Iterable[str], path: Path) -> None:
    present = set(rows[0]) if rows else set()
    missing = sorted(set(columns) - present)
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(missing)}")


def validate_file_names(expected: set[str], actual: set[str], label: str) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(f"Missing {label}: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unexpected {label}: {', '.join(extra)}")


def validate_mask_files(shared_config: dict[object, object]) -> None:
    expected = file_names_from_index(shared_config)
    data = shared_config["data"]
    if not isinstance(data, dict):
        raise ValueError("shared config data section must be a map")
    masks_dir = Path(str(data["masks_dir"]))
    by_stem: dict[str, list[Path]] = {}
    for path in masks_dir.iterdir():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            by_stem.setdefault(path.stem, []).append(path)
    duplicated = sorted(stem for stem, paths in by_stem.items() if len(paths) > 1)
    if duplicated:
        raise ValueError(f"Duplicated masks: {', '.join(duplicated)}")
    validate_file_names(expected, set(by_stem), "masks")
    for path in (paths[0] for paths in by_stem.values()):
        validate_binary_mask(path)


def validate_binary_mask(path: Path) -> None:
    image = Image.open(path).convert("L")
    values = np.unique(np.asarray(image))
    invalid_values = [int(value) for value in values if value not in (0, 255)]
    if invalid_values:
        preview = ", ".join(str(value) for value in invalid_values[:10])
        raise ValueError(f"mask must be binary black/white (0/255): {path}; found values: {preview}")


def validate_feature_index(shared_config: dict[object, object], feature_columns: list[str]) -> None:
    expected = file_names_from_index(shared_config)
    features = shared_config["features"]
    if not isinstance(features, dict):
        raise ValueError("shared config features section must be a map")
    path = Path(str(features["features_index_path"]))
    rows = read_csv_rows(path)
    validate_columns(rows, ["file_name", "weight", *feature_columns], path)
    validate_file_names(expected, {row["file_name"] for row in rows}, "feature rows")


def validate_split(shared_config: dict[object, object]) -> None:
    expected = file_names_from_index(shared_config)
    split = shared_config["split"]
    if not isinstance(split, dict):
        raise ValueError("shared config split section must be a map")
    path = Path(str(split["split_path"]))
    rows = read_csv_rows(path)
    validate_columns(
        rows,
        ["file_name", "weight", "weight_category", "weight_category_label", "fold"],
        path,
    )
    validate_file_names(expected, {row["file_name"] for row in rows}, "split rows")


def validate_shared_training_inputs(
    shared_config: dict[object, object], feature_columns: list[str] | None = None
) -> None:
    validate_mask_files(shared_config)
    validate_split(shared_config)
    if feature_columns is not None:
        validate_feature_index(shared_config, feature_columns)
