"""Validation for the curated Máscaras Válidas contract."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from buffalo_weight.reproduction_config import InputsContract

INDEX_COLUMNS = ["file_name", "farm", "weight_kg"]


@dataclass(frozen=True)
class ValidMask:
    file_name: str
    farm: str
    weight_kg: float
    path: Path


def validate_curated_inputs(contract: InputsContract) -> list[ValidMask]:
    """Validate curated inputs; for example, ``validate_curated_inputs(contract)``."""
    rows = _read_index(contract.mask_index_path)
    _validate_row_count(rows, contract.expected_mask_count, contract.mask_index_path)
    masks = [_valid_mask(row, contract.masks_dir) for row in rows]
    _validate_unique_names(masks)
    _validate_directory_files(masks, contract.masks_dir)
    _validate_pixels(masks)
    return sorted(masks, key=lambda mask: mask.file_name)


def input_hashes(contract: InputsContract) -> dict[str, str]:
    hashes = {"mask_index.csv": _file_hash(contract.mask_index_path)}
    if not contract.masks_dir.is_dir():
        raise ValueError(
            f"masks directory was {contract.masks_dir}; expected an existing directory"
        )
    for path in sorted(contract.masks_dir.iterdir()):
        if path.is_file():
            hashes[f"masks/{path.name}"] = _file_hash(path)
    return hashes


def _read_index(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"mask index was {path}; expected an existing CSV file")
    with path.open(newline="", encoding="utf-8") as index_file:
        reader = csv.DictReader(index_file)
        fields = list(reader.fieldnames or [])
        if fields != INDEX_COLUMNS:
            raise ValueError(f"mask index columns were {fields}; expected exactly {INDEX_COLUMNS}")
        return list(reader)


def _validate_row_count(rows: list[dict[str, str]], expected: int, path: Path) -> None:
    if len(rows) != expected:
        raise ValueError(
            f"mask index {path} had {len(rows)} rows; expected exactly {expected} Máscaras Válidas"
        )


def _valid_mask(row: dict[str, str], masks_dir: Path) -> ValidMask:
    file_name = row["file_name"]
    if Path(file_name).name != file_name or not file_name.endswith(".png"):
        raise ValueError(f"file_name was {file_name!r}; expected a basename ending in .png")
    farm = row["farm"].strip()
    if not farm:
        raise ValueError(f"farm was {row['farm']!r} for {file_name}; expected non-empty text")
    weight = _valid_weight(row["weight_kg"], file_name)
    return ValidMask(file_name, farm, weight, masks_dir / file_name)


def _valid_weight(value: str, file_name: str) -> float:
    try:
        weight = float(value)
    except ValueError as error:
        raise ValueError(
            f"weight_kg was {value!r} for {file_name}; expected a finite number greater than 0"
        ) from error
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(
            f"weight_kg was {value!r} for {file_name}; expected a finite number greater than 0"
        )
    return weight


def _validate_unique_names(masks: list[ValidMask]) -> None:
    names = [mask.file_name for mask in masks]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"file_name was repeated: {duplicated}; expected unique index names")


def _validate_directory_files(masks: list[ValidMask], masks_dir: Path) -> None:
    expected = {mask.file_name for mask in masks}
    actual = {path.name for path in masks_dir.iterdir() if path.is_file()}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing:
        raise ValueError(f"missing mask was {missing[0]}; expected exactly one indexed PNG")
    if extra:
        raise ValueError(f"extra mask was {extra[0]}; expected exactly the indexed PNG files")


def _validate_pixels(masks: list[ValidMask]) -> None:
    seen: dict[str, str] = {}
    for mask in masks:
        pixel_digest = _validated_pixel_digest(mask)
        duplicate = seen.get(pixel_digest)
        if duplicate is not None:
            raise ValueError(
                f"mask {mask.file_name} duplicated {duplicate}; expected pixel-wise unique masks"
            )
        seen[pixel_digest] = mask.file_name


def _validated_pixel_digest(mask: ValidMask) -> str:
    with Image.open(mask.path) as image:
        if image.mode != "L":
            raise ValueError(f"mask mode was {image.mode!r} for {mask.file_name}; expected grayscale L")
        pixels = np.asarray(image)
    values = np.unique(pixels)
    invalid = [int(value) for value in values if value not in (0, 255)]
    if invalid:
        raise ValueError(
            f"mask {mask.file_name} had pixel value {invalid[0]}; expected only pixel values 0 and 255"
        )
    shape = f"{pixels.shape[0]}x{pixels.shape[1]}:".encode()
    return hashlib.sha256(shape + pixels.tobytes()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
