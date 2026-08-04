"""Atomic promotion and downstream gate for human-confirmed feature evidence."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from buffalo_weight.feature_calculators import APPROVED_FEATURES
from buffalo_weight.feature_confirmation_contract import (
    decision_url,
    validate_confirmed_feature_contract,
)
from buffalo_weight.feature_confirmation_environment import (
    FeatureConfirmationEnvironment,
    LocalFeatureConfirmationEnvironment,
)
from buffalo_weight.feature_selection_artifacts import write_json_artifact
from buffalo_weight.feature_selection_manifest import (
    OFFICIAL_EXECUTION,
    OUTPUT_FILES,
    artifact_output_records,
    feature_selection_input_records,
    feature_selection_output_dir,
    feature_selection_status,
    validate_feature_selection_manifest,
)
from buffalo_weight.feature_selection_provenance import (
    FeatureSelectionProvenance,
    SystemFeatureSelectionProvenance,
)
from buffalo_weight.feature_selection_validation import (
    validate_feature_selection_artifacts,
    validate_feature_selection_evidence_files,
)
from buffalo_weight.hashing import sha256_file
from buffalo_weight.reproduction_config import ReportContract

CONFIRMED_OUTPUT_FILES = OUTPUT_FILES
CONFIRMED_VALIDATIONS = (
    "source_integrity", "schemas", "ordering", "sha256", "human_review",
    "candidate_subset", "official_execution", "clean_worktree",
)
CONFIRMED_MANIFEST_KEYS = {
    "manifest_version", "package_type", "revision", "status", "command",
    "source_commit", "source_feature_selection_manifest_sha256", "source_execution",
    "inputs", "report_sha256", "decision", "outputs", "validations",
}


def confirm_feature_selection(
    report_contract: ReportContract, human_contract_path: Path, reviewed_report_path: Path,
    dry_run: bool = False, environment: FeatureConfirmationEnvironment | None = None,
    provenance: FeatureSelectionProvenance | None = None,
) -> str:
    """Promote reviewed evidence; for example, ``dry_run=True`` performs no writes."""
    resolved_environment = environment or LocalFeatureConfirmationEnvironment()
    resolved_provenance = provenance or SystemFeatureSelectionProvenance()
    validation = _promotion_validation(
        report_contract, human_contract_path, reviewed_report_path, dry_run,
        resolved_environment, resolved_provenance,
    )
    if isinstance(validation, str):
        return validation
    human_contract, source_manifest = validation
    if dry_run:
        return "released"
    _publish_confirmed_package(
        report_contract, human_contract_path, reviewed_report_path,
        human_contract, source_manifest,
    )
    return "confirmed"


def _promotion_validation(
    report_contract: ReportContract, human_contract_path: Path, reviewed_report_path: Path,
    dry_run: bool, environment: FeatureConfirmationEnvironment,
    provenance: FeatureSelectionProvenance,
) -> tuple[dict[str, object], dict[str, object]] | str:
    try:
        return _validate_promotion(
            report_contract, human_contract_path, reviewed_report_path,
            environment, provenance,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        if dry_run:
            return f"blocked: {error}"
        raise


def baselines_gate_status(report_contract: ReportContract) -> str:
    """Classify the baseline gate; for example, tampering returns ``blocked``."""
    try:
        validate_confirmed_feature_package(report_contract)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "blocked"
    return "released"


def require_baselines_gate(report_contract: ReportContract) -> None:
    """Fail before baseline work; for example, a missing confirmation raises clearly."""
    try:
        validate_confirmed_feature_package(report_contract)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"confirmed feature gate was blocked by {error}; expected an intact compatible "
            "package at evidence/confirmed/feature_selection/v1"
        ) from error


def validate_confirmed_feature_package(report_contract: ReportContract) -> tuple[str, ...]:
    """Validate the gate package; for example, callers receive the frozen feature order."""
    package_dir = report_contract.confirmed_feature_selection_dir
    manifest = _read_mapping(package_dir / "manifest.json", "confirmed manifest")
    contract = _read_mapping(package_dir / "shared_feature_contract.json", "confirmed contract")
    report_path = package_dir / "feature_selection_report.md"
    validated = validate_confirmed_feature_contract(contract, report_path, APPROVED_FEATURES)
    _validate_confirmed_manifest(manifest, package_dir, validated, report_contract)
    validate_feature_selection_evidence_files(package_dir, APPROVED_FEATURES)
    selected = validated["selected_features"]
    if not isinstance(selected, list) or not all(isinstance(value, str) for value in selected):
        raise ValueError(f"selected_features was {selected!r}; expected a validated string list")
    return tuple(selected)


def _validate_promotion(
    report_contract: ReportContract, human_contract_path: Path, reviewed_report_path: Path,
    environment: FeatureConfirmationEnvironment, provenance: FeatureSelectionProvenance,
) -> tuple[dict[str, object], dict[str, object]]:
    _require_clean_worktree(environment)
    source_manifest = _validated_source_manifest(report_contract, provenance)
    human_contract = _read_mapping(human_contract_path, "human feature contract")
    validate_confirmed_feature_contract(human_contract, reviewed_report_path, APPROVED_FEATURES)
    return human_contract, source_manifest


def _require_clean_worktree(environment: FeatureConfirmationEnvironment) -> None:
    changes = environment.worktree_changes(Path(__file__).parents[2])
    if changes:
        raise ValueError(f"worktree changes were {changes!r}; expected a clean worktree")


def _validated_source_manifest(
    report_contract: ReportContract, provenance: FeatureSelectionProvenance,
) -> dict[str, object]:
    status = feature_selection_status(report_contract, provenance)
    if status != "reusable":
        raise ValueError(
            f"feature-selection stage status was {status!r}; expected reusable reviewed evidence"
        )
    source_dir = feature_selection_output_dir(report_contract)
    source_manifest = _read_mapping(source_dir / "manifest.json", "feature manifest")
    validate_feature_selection_manifest(source_manifest, source_dir)
    validate_feature_selection_artifacts(source_dir, APPROVED_FEATURES)
    if source_manifest.get("execution") != OFFICIAL_EXECUTION:
        raise ValueError(
            f"feature execution was {source_manifest.get('execution')!r}; "
            f"expected official execution {OFFICIAL_EXECUTION!r}"
        )
    return source_manifest


def _publish_confirmed_package(
    report_contract: ReportContract, human_contract_path: Path, reviewed_report_path: Path,
    human_contract: dict[str, object], source_manifest: dict[str, object],
) -> None:
    destination = report_contract.confirmed_feature_selection_dir
    if destination.exists() or destination.is_symlink():
        raise ValueError(
            f"confirmed feature destination was {destination}; expected absent v1 destination"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".feature-confirmation-", dir=destination.parent))
    try:
        _write_confirmed_package(
            temporary, report_contract, human_contract_path, reviewed_report_path,
            human_contract, source_manifest,
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_confirmed_package(
    output_dir: Path, report_contract: ReportContract, human_contract_path: Path,
    reviewed_report_path: Path, human_contract: dict[str, object],
    source_manifest: dict[str, object],
) -> None:
    source_dir = feature_selection_output_dir(report_contract)
    for name in CONFIRMED_OUTPUT_FILES:
        if name not in {"shared_feature_contract.json", "feature_selection_report.md"}:
            shutil.copy2(source_dir / name, output_dir / name)
    shutil.copy2(reviewed_report_path, output_dir / "feature_selection_report.md")
    shutil.copy2(human_contract_path, output_dir / "shared_feature_contract.json")
    validate_confirmed_feature_contract(
        human_contract, output_dir / "feature_selection_report.md", APPROVED_FEATURES,
    )
    validate_feature_selection_evidence_files(output_dir, APPROVED_FEATURES)
    manifest = _confirmed_manifest(output_dir, report_contract, human_contract, source_manifest)
    _validate_confirmed_manifest(manifest, output_dir, human_contract, report_contract)
    write_json_artifact(output_dir / "manifest.json", manifest)


def _confirmed_manifest(
    output_dir: Path, report_contract: ReportContract, human_contract: dict[str, object],
    source_manifest: dict[str, object],
) -> dict[str, object]:
    decision = {
        "url": decision_url(human_contract),
        "sha256": sha256_file(output_dir / "shared_feature_contract.json"),
    }
    manifest = {
        "manifest_version": 1, "package_type": "confirmed_evidence", "revision": 1,
        "status": "confirmed", "command": "python main.py confirm-features",
        "source_commit": source_manifest.get("source_commit"),
        "source_feature_selection_manifest_sha256": sha256_file(
            feature_selection_output_dir(report_contract) / "manifest.json"
        ),
        "source_execution": source_manifest.get("execution"),
        "inputs": source_manifest.get("inputs"),
        "report_sha256": sha256_file(output_dir / "feature_selection_report.md"),
        "decision": decision,
        "outputs": artifact_output_records(output_dir, CONFIRMED_OUTPUT_FILES),
        "validations": list(CONFIRMED_VALIDATIONS),
    }
    return manifest


def _validate_confirmed_manifest(
    manifest: dict[str, object], output_dir: Path, human_contract: dict[str, object],
    report_contract: ReportContract,
) -> None:
    _validate_manifest_fixed_fields(manifest)
    _validate_manifest_links(manifest, output_dir, human_contract)
    _validate_manifest_inputs(manifest, report_contract)
    outputs = artifact_output_records(output_dir, CONFIRMED_OUTPUT_FILES)
    if manifest.get("outputs") != outputs:
        raise ValueError(
            f"confirmed manifest outputs were {manifest.get('outputs')!r}; expected {outputs!r}"
        )


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
    _validate_source_identity(manifest)


def _validate_source_identity(manifest: dict[str, object]) -> None:
    source_commit = manifest.get("source_commit")
    source_hash = manifest.get("source_feature_selection_manifest_sha256")
    valid_commit = isinstance(source_commit, str) and len(source_commit) == 40
    valid_hash = isinstance(source_hash, str) and len(source_hash) == 64
    if not valid_commit or not valid_hash:
        raise ValueError(
            f"confirmed source identity was {source_commit!r}/{source_hash!r}; "
            "expected a 40-character commit and 64-character SHA-256"
        )


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
    decision = manifest.get("decision")
    expected_decision = {
        "url": decision_url(human_contract),
        "sha256": sha256_file(output_dir / "shared_feature_contract.json"),
    }
    expected_report = sha256_file(output_dir / "feature_selection_report.md")
    if decision != expected_decision or manifest.get("report_sha256") != expected_report:
        raise ValueError(
            f"confirmed manifest decision/report were {decision!r}/"
            f"{manifest.get('report_sha256')!r}; expected {expected_decision!r}/{expected_report!r}"
        )


def _read_mapping(path: Path, label: str) -> dict[str, object]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} at {path} was {loaded!r}; expected a JSON mapping")
    return loaded
