"""Promotion orchestration and downstream gate for confirmed expanded diagnostics evidence.

Reference: GitHub Issue #27.
"""

from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Literal

from buffalo_weight.diagnostic_confirmation_contract import (
    validate_confirmed_diagnostic_contract,
)
from buffalo_weight.diagnostic_confirmation_manifest import (
    read_json_mapping,
    validate_confirmed_diagnostic_manifest,
    validate_canonical_tables,
)
from buffalo_weight.diagnostic_confirmation_publication import (
    publish_confirmed_diagnostic_package,
)
from buffalo_weight.feature_confirmation import _GateInspection, _PromotionInspection
from buffalo_weight.feature_confirmation_environment import (
    FeatureConfirmationEnvironment,
    LocalFeatureConfirmationEnvironment,
)
from buffalo_weight.report_provenance import ReportProvenance, SystemReportProvenance
from buffalo_weight.reproduction_config import ReportContract


def confirm_diagnostics(
    report_contract: ReportContract,
    human_contract_path: Path,
    reviewed_report_path: Path,
    dry_run: bool = False,
    environment: FeatureConfirmationEnvironment | None = None,
    provenance: ReportProvenance | None = None,
) -> str:
    """Promote reviewed diagnostic evidence; for example, ``dry_run=True`` performs no writes.

    Example: ``confirm_diagnostics(contract, contract_p, report_p)`` promotes evidence.
    """
    resolved_environment = environment or LocalFeatureConfirmationEnvironment()
    resolved_provenance = provenance or SystemReportProvenance()
    validation = _promotion_validation(
        report_contract, human_contract_path, reviewed_report_path, dry_run,
        resolved_environment, resolved_provenance,
    )
    if validation.human_contract is None:
        return validation.blocked_message()
    if dry_run:
        return "released"
    publish_confirmed_diagnostic_package(
        report_contract, human_contract_path, reviewed_report_path, validation.human_contract,
        commit=resolved_provenance.repository_commit(),
    )
    return "confirmed"


def diagnostics_gate_status(report_contract: ReportContract) -> str:
    """Explain the diagnostic gate.

    Example: missing confirmation returns ``blocked: <reason>``.
    """
    return _inspect_diagnostics_gate(report_contract).message()


def require_diagnostics_gate(report_contract: ReportContract) -> None:
    """Fail before downstream work; for example, missing confirmation raises clearly."""
    inspection = _inspect_diagnostics_gate(report_contract)
    if inspection.state == "released":
        return
    raise ValueError(
        f"confirmed diagnostics gate was blocked by {inspection.reason}; expected an intact "
        "compatible package at evidence/confirmed/diagnostics/v1"
    )


def _inspect_diagnostics_gate(report_contract: ReportContract) -> _GateInspection:
    try:
        _validate_frozen_diagnostic_package(report_contract)
        return _GateInspection("released", None)
    except (OSError, ValueError, TypeError, JSONDecodeError) as error:
        return _GateInspection("blocked", str(error))


def _validate_frozen_diagnostic_package(report_contract: ReportContract) -> None:
    package_dir = report_contract.confirmed_diagnostics_dir
    contract_path = package_dir / "diagnostics_contract.json"
    human_contract = read_json_mapping(contract_path, "confirmed diagnostic contract")
    report_path = package_dir / "expanded_diagnostics_report.md"
    validate_confirmed_diagnostic_contract(human_contract, report_path)
    manifest = read_json_mapping(package_dir / "manifest.json", "confirmed diagnostic manifest")
    validate_confirmed_diagnostic_manifest(manifest, package_dir, human_contract, report_contract)


def _promotion_validation(
    report_contract: ReportContract,
    human_contract_path: Path,
    reviewed_report_path: Path,
    dry_run: bool,
    environment: FeatureConfirmationEnvironment,
    provenance: ReportProvenance,
) -> _PromotionInspection:
    try:
        human_contract = _validate_promotion(
            report_contract, human_contract_path, reviewed_report_path,
            environment, provenance,
        )
        return _PromotionInspection(human_contract, None)
    except (OSError, ValueError, TypeError, JSONDecodeError) as error:
        if dry_run:
            return _PromotionInspection(None, str(error))
        raise


def _validate_promotion(
    report_contract: ReportContract,
    human_contract_path: Path,
    reviewed_report_path: Path,
    environment: FeatureConfirmationEnvironment,
    provenance: ReportProvenance,
) -> dict[str, object]:
    _require_clean_worktree(environment)
    _validate_source_diagnostic_packages(report_contract)
    human_contract = read_json_mapping(human_contract_path, "human diagnostic contract")
    validate_confirmed_diagnostic_contract(human_contract, reviewed_report_path)
    return human_contract


def _require_clean_worktree(environment: FeatureConfirmationEnvironment) -> None:
    changes = environment.worktree_changes(Path(__file__).parents[2])
    if changes:
        raise ValueError(f"worktree changes were {changes!r}; expected a clean worktree")


def _validate_source_diagnostic_packages(report_contract: ReportContract) -> None:
    root = report_contract.artifacts_root / "diagnostics"
    for stage_name in ("descriptive", "learning_curves", "sensitivity"):
        stage_dir = root / stage_name
        manifest_path = stage_dir / "manifest.json"
        manifest = read_json_mapping(manifest_path, f"source {stage_name} manifest")
        if manifest.get("status") != "complete":
            raise ValueError(
                f"diagnostic stage {stage_name} status was {manifest.get('status')!r}; expected 'complete'"
            )
