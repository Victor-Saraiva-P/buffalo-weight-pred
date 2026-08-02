"""Atomic construction and safe cleanup of the inputs stage."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

from buffalo_weight.canonical_split import canonical_split_rows
from buffalo_weight.curated_inputs import ValidMask, validate_curated_inputs
from buffalo_weight.feature_calculators import APPROVED_FEATURES, calculate_mask_features
from buffalo_weight.input_schema import FEATURE_COLUMNS, SPLIT_COLUMNS
from buffalo_weight.inputs_manifest import complete_manifest, expected_identity, stage_status
from buffalo_weight.reproduction_config import ReportContract

STAGE_DESCENDANTS = {
    "inputs": ("inputs", "feature_selection", "baselines", "tuning", "diagnostics"),
    "feature_selection": ("feature_selection", "baselines", "tuning", "diagnostics"),
    "baselines": ("baselines", "tuning", "diagnostics"),
    "tuning": ("tuning", "diagnostics"),
    "diagnostics": ("diagnostics",),
}


def run_inputs_stage(contract: ReportContract, dry_run: bool = False) -> str:
    """Run the inputs stage; for example, ``run_inputs_stage(contract, dry_run=True)``."""
    status = stage_status(contract)
    if dry_run or status == "reusable":
        return status
    masks = validate_curated_inputs(contract.inputs)
    _build_atomic_snapshot(contract, masks)
    return "rebuilt"


def clean_reconstructible_stage(contract: ReportContract, stage: str) -> list[str]:
    """Clean a stage and descendants; for example, ``clean_reconstructible_stage(c, 'inputs')``."""
    descendants = STAGE_DESCENDANTS.get(stage)
    if descendants is None:
        expected = ", ".join(STAGE_DESCENDANTS)
        raise ValueError(f"stage was {stage!r}; expected one of: {expected}")
    removed: list[str] = []
    for name in descendants:
        path = contract.artifacts_root / name
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(name)
    return removed


def _build_atomic_snapshot(contract: ReportContract, masks: list[ValidMask]) -> None:
    root = contract.artifacts_root
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".inputs-", dir=root))
    identity = expected_identity(contract)
    try:
        _write_outputs(temporary, contract, masks)
        _validate_outputs(temporary, contract, masks)
        if expected_identity(contract) != identity:
            raise ValueError("inputs changed during the stage; expected an unchanged input snapshot")
        manifest = complete_manifest(contract, temporary, identity)
        (temporary / "manifest.json").write_text(_json_text(manifest))
        _replace_snapshot(temporary, contract.inputs_output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_outputs(output_dir: Path, contract: ReportContract, masks: list[ValidMask]) -> None:
    feature_rows = [_feature_row(mask, contract.inputs.canonical_long_side) for mask in masks]
    split_rows = canonical_split_rows(
        masks,
        contract.inputs.weight_category_count,
        contract.inputs.fold_count,
        contract.inputs.fold_seed,
    )
    _write_csv(output_dir / "feature_index.csv", FEATURE_COLUMNS, feature_rows)
    _write_csv(output_dir / "canonical_split.csv", SPLIT_COLUMNS, split_rows)


def _feature_row(mask: ValidMask, canonical_long_side: int) -> dict[str, str]:
    values = calculate_mask_features(mask.path, canonical_long_side)
    row = {
        "file_name": mask.file_name,
        "farm": mask.farm,
        "weight_kg": _format_number(mask.weight_kg),
    }
    row.update({name: _format_number(values[name]) for name in APPROVED_FEATURES})
    return row


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _validate_outputs(output_dir: Path, contract: ReportContract, masks: list[ValidMask]) -> None:
    expected_names = {mask.file_name for mask in masks}
    _validate_csv(output_dir / "feature_index.csv", FEATURE_COLUMNS, expected_names)
    split_rows = _validate_csv(output_dir / "canonical_split.csv", SPLIT_COLUMNS, expected_names)
    if len(split_rows) != contract.inputs.expected_mask_count:
        raise ValueError(
            f"canonical split rows were {len(split_rows)}; expected {contract.inputs.expected_mask_count}"
        )


def _validate_csv(path: Path, columns: list[str], names: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields, rows = list(reader.fieldnames or []), list(reader)
    actual_names = [row["file_name"] for row in rows]
    if fields != columns:
        raise ValueError(f"output columns were {fields}; expected exactly {columns}")
    if len(actual_names) != len(names) or set(actual_names) != names:
        raise ValueError(f"output names were {actual_names}; expected one row for each {sorted(names)}")
    return rows


def _replace_snapshot(temporary: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}-previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except BaseException:
        if backup.exists():
            os.replace(backup, destination)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _format_number(value: float) -> str:
    return f"{value:.6f}"


def _json_text(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"
