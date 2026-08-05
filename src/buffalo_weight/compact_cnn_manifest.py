"""Freshness identity and verifiable completion manifest for the compact CNN."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from buffalo_weight.compact_cnn_artifacts import (
    compact_cnn_output_records,
    validate_compact_cnn_output_tables,
)
from buffalo_weight.compact_cnn_provenance import CompactCnnProvenance
from buffalo_weight.compact_cnn_types import CompactCnnArtifactStatus, CompactCnnRecipe
from buffalo_weight.hashing import sha256_file
from buffalo_weight.reproduction_config import ReportContract


MANIFEST_VALIDATIONS = (
    "binary_mask_pixels", "canonical_split", "outer_fold_isolation",
    "inner_epoch_selection", "deterministic_cuda", "schemas", "ordering", "sha256",
)
MANIFEST_KEYS = {
    "manifest_version", "stage", "configuration", "recipe", "recipe_sha256",
    "dependencies", "inputs", "package_type", "revision", "status", "command",
    "source_commit", "execution", "decision_url", "reviewed_report_sha256",
    "validations", "outputs",
}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def compact_cnn_output_dir(contract: ReportContract) -> Path:
    """Locate the configuration artifact; for example, it nests beneath baselines."""
    output_dir = contract.artifacts_root / "baselines" / "compact_cnn"
    return output_dir


def compact_cnn_status(
    contract: ReportContract, recipe: CompactCnnRecipe, provenance: CompactCnnProvenance,
) -> CompactCnnArtifactStatus:
    """Classify freshness; for example, a changed recipe hash returns ``obsolete``."""
    output_dir = compact_cnn_output_dir(contract)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return "absent"
    try:
        manifest = json.loads(manifest_path.read_text())
        validate_compact_cnn_manifest(manifest, output_dir, contract, recipe, provenance)
        return "reusable"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "obsolete"


def build_compact_cnn_manifest(
    output_dir: Path, contract: ReportContract, recipe: CompactCnnRecipe,
    provenance: CompactCnnProvenance,
) -> dict[str, object]:
    """Complete the manifest; for example, output hashes make tampering obsolete."""
    manifest = compact_cnn_identity(contract, recipe, provenance)
    manifest.update({
        "package_type": "reconstructible_configuration", "revision": 1,
        "status": "complete", "command": "python main.py baselines",
        "source_commit": provenance.repository_commit(),
        "execution": provenance.compact_cnn_execution(), **_feature_gate_trace(contract),
        "validations": list(MANIFEST_VALIDATIONS),
        "outputs": compact_cnn_output_records(output_dir),
    })
    return manifest


def compact_cnn_identity(
    contract: ReportContract, recipe: CompactCnnRecipe, provenance: CompactCnnProvenance,
) -> dict[str, object]:
    """Fingerprint pertinent inputs; for example, output-only changes are excluded."""
    return {
        "manifest_version": 1, "stage": "baselines", "configuration": "compact_cnn",
        "recipe": recipe.as_mapping(), "recipe_sha256": provenance.compact_cnn_recipe_hash(),
        "dependencies": provenance.compact_cnn_dependencies(),
        "inputs": _input_records(contract),
    }


def validate_compact_cnn_manifest(
    manifest: object, output_dir: Path, contract: ReportContract,
    recipe: CompactCnnRecipe, provenance: CompactCnnProvenance,
) -> None:
    """Verify identity and outputs; for example, a modified CSV is rejected."""
    validated = _validated_manifest_mapping(manifest)
    _validate_identity(validated, compact_cnn_identity(contract, recipe, provenance))
    _validate_fixed_manifest_fields(validated)
    _validate_current_feature_trace(validated, contract)
    expected_outputs = compact_cnn_output_records(output_dir)
    if validated.get("outputs") != expected_outputs:
        raise ValueError(
            f"compact CNN outputs were {validated.get('outputs')!r}; "
            f"expected {expected_outputs!r}"
        )
    validate_compact_cnn_output_tables(output_dir, contract.inputs.expected_mask_count)


def write_manifest_last(path: Path, manifest: dict[str, object]) -> None:
    """Publish completion last; for example, interruption cannot leave a reusable stage."""
    serialized = json.dumps(manifest, indent=2, sort_keys=True)
    path.write_text(f"{serialized}\n")


def _validated_manifest_mapping(manifest: object) -> dict[str, object]:
    if not isinstance(manifest, dict):
        raise ValueError(f"compact CNN manifest was {manifest!r}; expected a JSON mapping")
    if set(manifest) != MANIFEST_KEYS:
        raise ValueError(
            f"compact CNN manifest keys were {sorted(manifest)!r}; "
            f"expected exactly {sorted(MANIFEST_KEYS)!r}"
        )
    return manifest


def _validate_identity(
    manifest: dict[str, object], expected: dict[str, object],
) -> None:
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"compact CNN identity was {actual!r}; expected {expected!r}")


def _input_records(contract: ReportContract) -> dict[str, dict[str, object]]:
    paths = {
        "inputs_manifest": contract.inputs_output_dir / "manifest.json",
        "canonical_split": contract.inputs_output_dir / "canonical_split.csv",
    }
    records = {name: _input_record(path) for name, path in paths.items()}
    records["referenced_masks"] = _referenced_mask_record(contract)
    return records


def _input_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"baseline input was {path}; expected an existing current artifact")
    record: dict[str, object] = {"path": str(path), "sha256": sha256_file(path)}
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.reader(source))
        record.update({"schema": rows[0], "row_count": len(rows) - 1})
    return record


def _referenced_mask_record(contract: ReportContract) -> dict[str, object]:
    split_path = contract.inputs_output_dir / "canonical_split.csv"
    with split_path.open(newline="", encoding="utf-8") as source:
        names = sorted(row["file_name"] for row in csv.DictReader(source))
    digest = hashlib.sha256()
    for name in names:
        path = contract.inputs.masks_dir / name
        if not path.is_file():
            raise ValueError(f"referenced mask was {path}; expected an indexed PNG file")
        digest.update(name.encode())
        digest.update(sha256_file(path).encode())
    return {"path": str(contract.inputs.masks_dir), "file_count": len(names),
            "sha256": digest.hexdigest()}


def _feature_gate_trace(contract: ReportContract) -> dict[str, str]:
    contract_path = contract.confirmed_feature_selection_dir / "shared_feature_contract.json"
    report_path = contract.confirmed_feature_selection_dir / "feature_selection_report.md"
    confirmed = json.loads(contract_path.read_text())
    decision = confirmed.get("human_decision") if isinstance(confirmed, dict) else None
    decision_url = decision.get("decision_url") if isinstance(decision, dict) else None
    if not isinstance(decision_url, str) or not decision_url:
        raise ValueError(f"feature decision URL was {decision_url!r}; expected non-empty text")
    return {"decision_url": decision_url, "reviewed_report_sha256": sha256_file(report_path)}


def _validate_fixed_manifest_fields(manifest: dict[str, object]) -> None:
    expected = ("reconstructible_configuration", 1, "complete", "python main.py baselines")
    actual = (manifest.get("package_type"), manifest.get("revision"),
              manifest.get("status"), manifest.get("command"))
    if actual != expected:
        raise ValueError(f"compact CNN manifest fields were {actual!r}; expected {expected!r}")
    _validate_manifest_trace(manifest)
    execution = manifest.get("execution")
    if not isinstance(execution, dict) or execution.get("device") != "cuda":
        raise ValueError(f"compact CNN execution was {execution!r}; expected CUDA audit fields")


def _validate_manifest_trace(manifest: dict[str, object]) -> None:
    source_commit = manifest.get("source_commit")
    report_hash = manifest.get("reviewed_report_sha256")
    decision_url = manifest.get("decision_url")
    valid_hashes = (isinstance(source_commit, str) and HEX_COMMIT.fullmatch(source_commit)
                    and isinstance(report_hash, str) and HEX_SHA256.fullmatch(report_hash))
    if not valid_hashes or not isinstance(decision_url, str) or not decision_url:
        actual = (source_commit, report_hash, decision_url)
        raise ValueError(f"compact CNN trace was {actual!r}; expected commit, report hash and URL")
    if manifest.get("validations") != list(MANIFEST_VALIDATIONS):
        raise ValueError(
            f"compact CNN validations were {manifest.get('validations')!r}; "
            f"expected {list(MANIFEST_VALIDATIONS)!r}"
        )


def _validate_current_feature_trace(
    manifest: dict[str, object], contract: ReportContract,
) -> None:
    expected = _feature_gate_trace(contract)
    actual = {name: manifest.get(name) for name in expected}
    if actual != expected:
        raise ValueError(
            f"compact CNN feature trace was {actual!r}; expected current gate {expected!r}"
        )
