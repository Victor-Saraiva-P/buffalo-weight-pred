"""Freshness and integrity manifests for the inputs stage."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from buffalo_weight.curated_inputs import input_hashes
from buffalo_weight.input_schema import OUTPUT_SCHEMAS
from buffalo_weight.reproduction_config import ReportContract, contract_identity

MANIFEST_VERSION = 1
OUTPUT_FILES = ("feature_index.csv", "canonical_split.csv")


def expected_identity(contract: ReportContract) -> dict[str, object]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "stage": "inputs",
        "contract": contract_identity(contract),
        "recipe_sha256": recipe_hash(),
        "dependencies": _dependency_versions(),
        "inputs": input_hashes(contract.inputs),
    }


def stage_status(contract: ReportContract) -> str:
    output_dir = contract.inputs_output_dir
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return "absent"
    try:
        manifest = json.loads(manifest_path.read_text())
        return "reusable" if _manifest_is_current(manifest, contract) else "obsolete"
    except (OSError, ValueError, TypeError):
        return "obsolete"


def complete_manifest(
    contract: ReportContract, output_dir: Path, identity: dict[str, object]
) -> dict[str, object]:
    manifest = identity.copy()
    manifest.update(
        {
            "package_type": "reconstructible_stage",
            "revision": 1,
            "status": "complete",
            "source_commit": _source_commit(),
            "command": "python main.py inputs",
            "row_count": contract.inputs.expected_mask_count,
            "outputs": _output_records(output_dir),
            "validations": _validation_names(),
        }
    )
    return manifest


def _output_records(output_dir: Path) -> dict[str, dict[str, object]]:
    return {
        name: {
            "sha256": file_hash(output_dir / name),
            "rows": _csv_rows(output_dir / name),
            "columns": _csv_columns(output_dir / name),
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


def recipe_hash() -> str:
    module_names = (
        "canonical_split.py",
        "curated_inputs.py",
        "input_schema.py",
        "inputs_manifest.py",
        "report_inputs.py",
        "reproduction_config.py",
    )
    source_root = Path(__file__).parent
    calculators = sorted((source_root / "feature_calculators").glob("*.py"))
    return _combined_hash([source_root / name for name in module_names] + calculators)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_is_current(manifest: object, contract: ReportContract) -> bool:
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        return False
    identity = expected_identity(contract)
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
        record.get("sha256") == file_hash(path)
        and record.get("rows") == expected_rows == _csv_rows(path)
        and record.get("columns") == OUTPUT_SCHEMAS[name] == _csv_columns(path)
    )


def _csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as csv_file:
        return max(sum(1 for _ in csv_file) - 1, 0)


def _csv_columns(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as csv_file:
        return csv_file.readline().rstrip("\r\n").split(",")


def _combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(Path(__file__).parent).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    distributions = ("numpy", "Pillow", "scipy", "scikit-learn", "PyYAML")
    return {name: importlib.metadata.version(name) for name in distributions}


def _source_commit() -> str:
    repository_root = Path(__file__).parents[2]
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
