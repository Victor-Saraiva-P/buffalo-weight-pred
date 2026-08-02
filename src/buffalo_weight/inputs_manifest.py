"""Freshness and integrity manifests for the inputs stage."""

from __future__ import annotations

import json
from pathlib import Path

from buffalo_weight.csv_io import csv_columns, csv_row_count
from buffalo_weight.curated_inputs import input_hashes
from buffalo_weight.hashing import sha256_file
from buffalo_weight.input_schema import OUTPUT_SCHEMAS
from buffalo_weight.report_provenance import ReportProvenance
from buffalo_weight.reproduction_config import ReportContract, contract_identity

MANIFEST_VERSION = 1
OUTPUT_FILES = ("feature_index.csv", "canonical_split.csv")


def expected_identity(
    contract: ReportContract, provenance: ReportProvenance
) -> dict[str, object]:
    """Build identity; for example, ``expected_identity(contract, provenance)`` fingerprints inputs."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "stage": "inputs",
        "contract": contract_identity(contract),
        "recipe_sha256": provenance.inputs_recipe_hash(),
        "dependencies": provenance.dependencies(),
        "inputs": input_hashes(contract.inputs),
    }


def stage_status(contract: ReportContract, provenance: ReportProvenance) -> str:
    """Classify freshness; for example, ``stage_status(contract, provenance)`` may be reusable."""
    output_dir = contract.inputs_output_dir
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return "absent"
    try:
        manifest = json.loads(manifest_path.read_text())
        current = _manifest_is_current(manifest, contract, provenance)
        return "reusable" if current else "obsolete"
    except (OSError, ValueError, TypeError):
        return "obsolete"


def complete_manifest(
    contract: ReportContract,
    output_dir: Path,
    identity: dict[str, object],
    source_commit: str,
) -> dict[str, object]:
    """Describe outputs; for example, ``complete_manifest(c, root, identity, commit)``."""
    manifest = identity.copy()
    manifest.update(_completion_fields(contract, output_dir, source_commit))
    return manifest


def _completion_fields(
    contract: ReportContract, output_dir: Path, source_commit: str
) -> dict[str, object]:
    return {
        "package_type": "reconstructible_stage", "revision": 1,
        "status": "complete", "source_commit": source_commit,
        "command": "python main.py inputs",
        "row_count": contract.inputs.expected_mask_count,
        "outputs": _output_records(output_dir), "validations": _validation_names(),
    }


def validate_complete_manifest(
    manifest: dict[str, object], output_dir: Path, expected_rows: int
) -> None:
    """Validate hashes and schemas before publication.

    Example: ``validate_complete_manifest(manifest, temp_dir, 132)`` checks a snapshot.
    """
    outputs = manifest.get("outputs")
    if manifest.get("row_count") != expected_rows or not isinstance(outputs, dict):
        raise ValueError(
            f"manifest row_count was {manifest.get('row_count')!r}; expected {expected_rows}"
        )
    for name in OUTPUT_FILES:
        record = outputs.get(name)
        if not isinstance(record, dict) or not _output_record_matches(
            record, output_dir / name, name, expected_rows
        ):
            raise ValueError(f"manifest output was {record!r} for {name}; expected valid hash/schema/count")


def _output_records(output_dir: Path) -> dict[str, dict[str, object]]:
    return {
        name: {
            "sha256": sha256_file(output_dir / name),
            "rows": csv_row_count(output_dir / name),
            "columns": csv_columns(output_dir / name),
        }
        for name in OUTPUT_FILES
    }


def _validation_names() -> list[str]:
    return [
        "schemas",
        "sha256",
        "row_counts",
        "one_to_one_mask_correspondence",
        "canonical_fold_distribution",
    ]


def _manifest_is_current(
    manifest: object, contract: ReportContract, provenance: ReportProvenance
) -> bool:
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        return False
    identity = expected_identity(contract, provenance)
    if any(manifest.get(key) != value for key, value in identity.items()):
        return False
    if manifest.get("row_count") != contract.inputs.expected_mask_count:
        return False
    outputs = manifest.get("outputs")
    return isinstance(outputs, dict) and _outputs_match(outputs, contract)


def _outputs_match(outputs: dict[object, object], contract: ReportContract) -> bool:
    for name in OUTPUT_FILES:
        record = outputs.get(name)
        path = contract.inputs_output_dir / name
        if not isinstance(record, dict) or not path.is_file():
            return False
        if not _output_record_matches(record, path, name, contract.inputs.expected_mask_count):
            return False
    return True


def _output_record_matches(
    record: dict[object, object], path: Path, name: str, expected_rows: int
) -> bool:
    return (
        record.get("sha256") == sha256_file(path)
        and record.get("rows") == expected_rows == csv_row_count(path)
        and record.get("columns") == OUTPUT_SCHEMAS[name] == csv_columns(path)
    )
