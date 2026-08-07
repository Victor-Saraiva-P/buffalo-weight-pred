"""Manifest construction and validation for confirmed approach selection evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path

from buffalo_weight.approach_confirmation_contract import (
    decision_url,
    validate_confirmed_approach_contract,
)
from buffalo_weight.baseline_comparison_manifest import (
    baseline_comparison_input_records,
)
from buffalo_weight.feature_confirmation_manifest import read_json_mapping
from buffalo_weight.feature_selection_manifest import artifact_output_records
from buffalo_weight.hashing import sha256_file
from buffalo_weight.reproduction_config import ReportContract

SOURCE_MANIFEST_NAME = "source_baseline_comparison_manifest.json"
CONFIRMED_OUTPUT_FILES = (
    "selected_approach.json", "approach_selection_report.md", "baseline_metrics.csv",
    SOURCE_MANIFEST_NAME,
)
CONFIRMED_VALIDATIONS = (
    "source_integrity", "schemas", "ordering", "sha256", "human_review",
    "compatible_approach", "clean_worktree",
)
CONFIRMED_MANIFEST_KEYS = {
    "manifest_version", "package_type", "revision", "status", "command",
    "source_commit", "source_baseline_comparison_manifest_sha256",
    "inputs", "report_sha256", "decision", "selected_approach", "outputs", "validations",
}
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_confirmed_approach_package(report_contract: ReportContract) -> tuple[str, str, int]:
    """Validate the approach gate package; for example, callers receive confirmed choice."""
    return _validate_confirmed_approach_package(report_contract, require_current_inputs=True)


def validate_frozen_approach_contract(report_contract: ReportContract) -> tuple[str, str, int]:
    """Validate frozen approach evidence; for example, downstream stages check contract integrity."""
    return _validate_confirmed_approach_package(report_contract, require_current_inputs=False)


def _validate_confirmed_approach_package(
    report_contract: ReportContract, require_current_inputs: bool,
) -> tuple[str, str, int]:
    package_dir = report_contract.confirmed_approach_selection_dir
    manifest = read_json_mapping(package_dir / "manifest.json", "confirmed approach manifest")
    contract = read_json_mapping(
        package_dir / "selected_approach.json", "confirmed approach contract",
    )
    report_path = package_dir / "approach_selection_report.md"
    validated = validate_confirmed_approach_contract(contract, report_path)
    validate_confirmed_approach_manifest(
        manifest, package_dir, validated, report_contract, require_current_inputs,
    )
    return _validated_choice(validated)


def build_confirmed_approach_manifest(
    output_dir: Path, report_contract: ReportContract, human_contract: dict[str, object],
) -> dict[str, object]:
    """Build the root approach manifest; for example, source identity comes from the copied record."""
    source_path = output_dir / SOURCE_MANIFEST_NAME
    source_manifest = read_json_mapping(source_path, "source comparison manifest")
    manifest = _confirmed_manifest_base(output_dir, source_manifest)
    manifest.update({
        "source_baseline_comparison_manifest_sha256": sha256_file(source_path),
        "decision": _decision_record(output_dir, human_contract),
        "selected_approach": _selected_approach_record(human_contract),
        "outputs": artifact_output_records(output_dir, CONFIRMED_OUTPUT_FILES),
    })
    return manifest


def validate_confirmed_approach_manifest(
    manifest: dict[str, object], output_dir: Path, human_contract: dict[str, object],
    report_contract: ReportContract, require_current_inputs: bool = True,
) -> None:
    """Validate the root approach manifest; for example, copied source provenance is rehashed."""
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


def _confirmed_manifest_base(
    output_dir: Path, source_manifest: dict[str, object],
) -> dict[str, object]:
    return {
        "manifest_version": 1, "package_type": "confirmed_evidence", "revision": 1,
        "status": "confirmed", "command": "python main.py confirm-approach",
        "source_commit": source_manifest.get("source_commit"),
        "inputs": source_manifest.get("inputs"),
        "report_sha256": sha256_file(output_dir / "approach_selection_report.md"),
        "validations": list(CONFIRMED_VALIDATIONS),
    }


def _decision_record(
    output_dir: Path, human_contract: dict[str, object],
) -> dict[str, str]:
    return {
        "url": decision_url(human_contract),
        "sha256": sha256_file(output_dir / "selected_approach.json"),
    }


def _selected_approach_record(
    human_contract: dict[str, object],
) -> dict[str, object]:
    return {
        "approach": human_contract["selected_approach"],
        "baseline_configuration": human_contract["baseline_configuration"],
        "maximum_tuning_variations": human_contract["maximum_tuning_variations"],
    }


def _validate_manifest_fixed_fields(manifest: dict[str, object]) -> None:
    if set(manifest) != CONFIRMED_MANIFEST_KEYS:
        raise ValueError(
            f"confirmed manifest keys were {sorted(manifest)!r}; expected exactly "
            f"{sorted(CONFIRMED_MANIFEST_KEYS)!r}"
        )
    fixed = (manifest.get("manifest_version"), manifest.get("package_type"),
             manifest.get("revision"), manifest.get("status"), manifest.get("command"),
             manifest.get("validations"))
    expected = (1, "confirmed_evidence", 1, "confirmed", "python main.py confirm-approach",
                list(CONFIRMED_VALIDATIONS))
    if fixed != expected:
        raise ValueError(f"confirmed manifest fixed fields were {fixed!r}; expected {expected!r}")


def _validate_source_identity(manifest: dict[str, object], output_dir: Path) -> None:
    source_path = output_dir / SOURCE_MANIFEST_NAME
    source_manifest = read_json_mapping(source_path, "source comparison manifest")
    source_commit = manifest.get("source_commit")
    source_hash = manifest.get("source_baseline_comparison_manifest_sha256")
    expected = (
        source_manifest.get("source_commit"), sha256_file(source_path),
    )
    actual = (source_commit, source_hash)
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
    expected_inputs = baseline_comparison_input_records(report_contract)
    recorded_inputs = manifest.get("inputs")
    if recorded_inputs != expected_inputs:
        raise ValueError(
            f"confirmed manifest inputs were {recorded_inputs!r}; "
            f"expected current inputs {expected_inputs!r}"
        )


def _validate_manifest_links(
    manifest: dict[str, object], output_dir: Path, human_contract: dict[str, object],
) -> None:
    expected_decision = _decision_record(output_dir, human_contract)
    expected_report = sha256_file(output_dir / "approach_selection_report.md")
    expected_selected = _selected_approach_record(human_contract)
    actual = (
        manifest.get("decision"), manifest.get("report_sha256"), manifest.get("selected_approach"),
    )
    expected = (expected_decision, expected_report, expected_selected)
    if actual != expected:
        raise ValueError(
            f"confirmed manifest decision/report/selected were {actual!r}; expected {expected!r}"
        )


def _validated_choice(validated: dict[str, object]) -> tuple[str, str, int]:
    approach = str(validated["selected_approach"])
    config = str(validated["baseline_configuration"])
    budget = int(str(validated["maximum_tuning_variations"]))
    return approach, config, budget
