"""Manifest construction and validation for confirmed feature evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path

from buffalo_weight.feature_calculators import APPROVED_FEATURES
from buffalo_weight.feature_confirmation_contract import (
    decision_url,
    validate_confirmed_feature_contract,
)
from buffalo_weight.feature_selection_manifest import (
    OFFICIAL_EXECUTION,
    OUTPUT_FILES,
    artifact_output_records,
    feature_selection_input_records,
)
from buffalo_weight.feature_selection_validation import validate_feature_selection_evidence_files
from buffalo_weight.hashing import sha256_file
from buffalo_weight.reproduction_config import ReportContract

SOURCE_MANIFEST_NAME = "source_feature_selection_manifest.json"
CONFIRMED_OUTPUT_FILES = (*OUTPUT_FILES, SOURCE_MANIFEST_NAME)
CONFIRMED_VALIDATIONS = (
    "source_integrity", "schemas", "ordering", "sha256", "human_review",
    "candidate_subset", "official_execution", "clean_worktree",
)
CONFIRMED_MANIFEST_KEYS = {
    "manifest_version", "package_type", "revision", "status", "command",
    "source_commit", "source_feature_selection_manifest_sha256", "source_execution",
    "inputs", "report_sha256", "decision", "outputs", "validations",
}
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_confirmed_feature_package(report_contract: ReportContract) -> tuple[str, ...]:
    """Validate the gate package; for example, callers receive the frozen feature order."""
    features = _validate_confirmed_feature_package(report_contract, require_current_inputs=True)
    return features


def validate_frozen_feature_contract(report_contract: ReportContract) -> tuple[str, ...]:
    """Validate frozen evidence; for example, baselines fingerprint only consumed columns."""
    features = _validate_confirmed_feature_package(report_contract, require_current_inputs=False)
    return features


def _validate_confirmed_feature_package(
    report_contract: ReportContract, require_current_inputs: bool,
) -> tuple[str, ...]:
    package_dir = report_contract.confirmed_feature_selection_dir
    manifest = read_json_mapping(package_dir / "manifest.json", "confirmed manifest")
    contract = read_json_mapping(
        package_dir / "shared_feature_contract.json", "confirmed contract",
    )
    report_path = package_dir / "feature_selection_report.md"
    validated = validate_confirmed_feature_contract(contract, report_path, APPROVED_FEATURES)
    validate_confirmed_manifest(
        manifest, package_dir, validated, report_contract, require_current_inputs,
    )
    validate_feature_selection_evidence_files(package_dir, APPROVED_FEATURES)
    return _validated_selection(validated.get("selected_features"))


def build_confirmed_manifest(
    output_dir: Path, report_contract: ReportContract, human_contract: dict[str, object],
) -> dict[str, object]:
    """Build the root manifest; for example, its source identity comes from the copied record."""
    source_path = output_dir / SOURCE_MANIFEST_NAME
    source_manifest = read_json_mapping(source_path, "source feature manifest")
    manifest = _confirmed_manifest_base(output_dir, source_manifest)
    manifest.update({
        "source_feature_selection_manifest_sha256": sha256_file(source_path),
        "decision": _decision_record(output_dir, human_contract),
        "outputs": artifact_output_records(output_dir, CONFIRMED_OUTPUT_FILES),
    })
    return manifest


def validate_confirmed_manifest(
    manifest: dict[str, object], output_dir: Path, human_contract: dict[str, object],
    report_contract: ReportContract, require_current_inputs: bool = True,
) -> None:
    """Validate the root manifest; for example, copied source provenance is rehashed."""
    _validate_manifest_fixed_fields(manifest)
    _validate_source_identity(manifest, output_dir)
    _validate_manifest_links(manifest, output_dir, human_contract)
    if require_current_inputs:
        _validate_manifest_inputs(manifest, report_contract)
    expected_outputs = artifact_output_records(output_dir, CONFIRMED_OUTPUT_FILES)
    if manifest.get("outputs") != expected_outputs:
        raise ValueError(
            f"confirmed manifest outputs were {manifest.get('outputs')!r}; "
            f"expected {expected_outputs!r}"
        )


def read_json_mapping(path: Path, label: str) -> dict[str, object]:
    """Read one JSON mapping; for example, malformed human contracts are rejected."""
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} at {path} was {loaded!r}; expected a JSON mapping")
    return loaded


def _confirmed_manifest_base(
    output_dir: Path, source_manifest: dict[str, object],
) -> dict[str, object]:
    return {
        "manifest_version": 1, "package_type": "confirmed_evidence", "revision": 1,
        "status": "confirmed", "command": "python main.py confirm-features",
        "source_commit": source_manifest.get("source_commit"),
        "source_execution": source_manifest.get("execution"),
        "inputs": source_manifest.get("inputs"),
        "report_sha256": sha256_file(output_dir / "feature_selection_report.md"),
        "validations": list(CONFIRMED_VALIDATIONS),
    }


def _decision_record(
    output_dir: Path, human_contract: dict[str, object],
) -> dict[str, str]:
    return {
        "url": decision_url(human_contract),
        "sha256": sha256_file(output_dir / "shared_feature_contract.json"),
    }


def _validate_manifest_fixed_fields(manifest: dict[str, object]) -> None:
    if set(manifest) != CONFIRMED_MANIFEST_KEYS:
        raise ValueError(
            f"confirmed manifest keys were {sorted(manifest)!r}; expected exactly "
            f"{sorted(CONFIRMED_MANIFEST_KEYS)!r}"
        )
    fixed = (manifest.get("manifest_version"), manifest.get("package_type"),
             manifest.get("revision"), manifest.get("status"), manifest.get("command"),
             manifest.get("source_execution"), manifest.get("validations"))
    expected = (1, "confirmed_evidence", 1, "confirmed", "python main.py confirm-features",
                OFFICIAL_EXECUTION, list(CONFIRMED_VALIDATIONS))
    if fixed != expected:
        raise ValueError(f"confirmed manifest fixed fields were {fixed!r}; expected {expected!r}")


def _validate_source_identity(manifest: dict[str, object], output_dir: Path) -> None:
    source_path = output_dir / SOURCE_MANIFEST_NAME
    source_manifest = read_json_mapping(source_path, "source feature manifest")
    source_commit = manifest.get("source_commit")
    source_hash = manifest.get("source_feature_selection_manifest_sha256")
    expected = (
        source_manifest.get("source_commit"), sha256_file(source_path),
        source_manifest.get("execution"),
    )
    actual = (source_commit, source_hash, manifest.get("source_execution"))
    if actual != expected or not _valid_source_hashes(source_commit, source_hash):
        raise ValueError(
            f"confirmed source identity was {actual!r}; expected {expected!r} with hex hashes"
        )


def _valid_source_hashes(source_commit: object, source_hash: object) -> bool:
    commit_valid = isinstance(source_commit, str) and GIT_SHA.fullmatch(source_commit) is not None
    hash_valid = isinstance(source_hash, str) and SHA256.fullmatch(source_hash) is not None
    return commit_valid and hash_valid


def _validate_manifest_inputs(
    manifest: dict[str, object], report_contract: ReportContract,
) -> None:
    expected_inputs = feature_selection_input_records(report_contract)
    if manifest.get("inputs") != expected_inputs:
        raise ValueError(
            f"confirmed manifest inputs were {manifest.get('inputs')!r}; "
            f"expected current inputs {expected_inputs!r}"
        )


def _validate_manifest_links(
    manifest: dict[str, object], output_dir: Path, human_contract: dict[str, object],
) -> None:
    expected_decision = _decision_record(output_dir, human_contract)
    expected_report = sha256_file(output_dir / "feature_selection_report.md")
    actual = (manifest.get("decision"), manifest.get("report_sha256"))
    expected = (expected_decision, expected_report)
    if actual != expected:
        raise ValueError(
            f"confirmed manifest decision/report were {actual!r}; expected {expected!r}"
        )


def _validated_selection(candidate: object) -> tuple[str, ...]:
    if not isinstance(candidate, list) or not all(isinstance(value, str) for value in candidate):
        raise ValueError(f"selected_features was {candidate!r}; expected a validated string list")
    return tuple(candidate)
