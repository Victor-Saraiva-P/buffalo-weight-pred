"""Promotion orchestration and downstream gate for confirmed approach selection evidence."""

from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Literal

from buffalo_weight.approach_confirmation_contract import validate_confirmed_approach_contract
from buffalo_weight.approach_confirmation_manifest import (
    read_json_mapping,
    validate_frozen_approach_contract,
)
from buffalo_weight.approach_confirmation_publication import publish_confirmed_approach_package
from buffalo_weight.baseline_comparison_manifest import (
    approach_selection_output_dir,
    baseline_comparison_status,
    validate_baseline_comparison_manifest,
)
from buffalo_weight.baseline_comparison_provenance import (
    BaselineComparisonProvenance,
    SystemBaselineComparisonProvenance,
)
from buffalo_weight.feature_confirmation import _GateInspection, _PromotionInspection
from buffalo_weight.feature_confirmation_environment import (
    FeatureConfirmationEnvironment,
    LocalFeatureConfirmationEnvironment,
)
from buffalo_weight.reproduction_config import ReportContract


def confirm_approach_selection(
    report_contract: ReportContract, human_contract_path: Path, reviewed_report_path: Path,
    dry_run: bool = False, environment: FeatureConfirmationEnvironment | None = None,
    provenance: BaselineComparisonProvenance | None = None,
) -> str:
    """Promote reviewed approach evidence; for example, ``dry_run=True`` performs no writes."""
    resolved_environment = environment or LocalFeatureConfirmationEnvironment()
    resolved_provenance = provenance or SystemBaselineComparisonProvenance()
    validation = _promotion_validation(
        report_contract, human_contract_path, reviewed_report_path, dry_run,
        resolved_environment, resolved_provenance,
    )
    if validation.human_contract is None:
        return validation.blocked_message()
    if dry_run:
        return "released"
    publish_confirmed_approach_package(
        report_contract, human_contract_path, reviewed_report_path, validation.human_contract,
    )
    return "confirmed"


def approach_gate_status(report_contract: ReportContract) -> str:
    """Explain the approach selection gate.

    Example: tampering returns ``blocked: <reason>``.
    """
    return _inspect_approach_gate(report_contract).message()


def require_approach_gate(report_contract: ReportContract) -> tuple[str, str, int]:
    """Fail before tuning or diagnostic work; for example, missing confirmation raises clearly."""
    inspection, choice = _inspect_approach_gate_with_choice(report_contract)
    if inspection.state == "released" and choice is not None:
        return choice
    raise ValueError(
        f"confirmed approach gate was blocked by {inspection.reason}; expected an intact "
        "compatible package at evidence/confirmed/approach_selection/v1"
    )


def _inspect_approach_gate(report_contract: ReportContract) -> _GateInspection:
    inspection, _choice = _inspect_approach_gate_with_choice(report_contract)
    gate_state = inspection.state
    gate_reason = inspection.reason
    return _GateInspection(gate_state, gate_reason)


def _inspect_approach_gate_with_choice(
    report_contract: ReportContract,
) -> tuple[_GateInspection, tuple[str, str, int] | None]:
    try:
        choice = validate_frozen_approach_contract(report_contract)
        return _GateInspection("released", None), choice
    except (OSError, ValueError, TypeError, JSONDecodeError) as error:
        return _GateInspection("blocked", str(error)), None


def _promotion_validation(
    report_contract: ReportContract, human_contract_path: Path, reviewed_report_path: Path,
    dry_run: bool, environment: FeatureConfirmationEnvironment,
    provenance: BaselineComparisonProvenance,
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
    report_contract: ReportContract, human_contract_path: Path, reviewed_report_path: Path,
    environment: FeatureConfirmationEnvironment, provenance: BaselineComparisonProvenance,
) -> dict[str, object]:
    _require_clean_worktree(environment)
    _validate_source_package(report_contract, provenance)
    human_contract = read_json_mapping(human_contract_path, "human approach contract")
    validate_confirmed_approach_contract(human_contract, reviewed_report_path)
    return human_contract


def _require_clean_worktree(environment: FeatureConfirmationEnvironment) -> None:
    changes = environment.worktree_changes(Path(__file__).parents[2])
    if changes:
        raise ValueError(f"worktree changes were {changes!r}; expected a clean worktree")


def _validate_source_package(
    report_contract: ReportContract, provenance: BaselineComparisonProvenance,
) -> None:
    status = baseline_comparison_status(report_contract, provenance)
    if status != "reusable":
        raise ValueError(
            f"baseline-comparison stage status was {status!r}; expected reusable reviewed evidence"
        )
    source_dir = approach_selection_output_dir(report_contract)
    manifest = read_json_mapping(source_dir / "manifest.json", "baseline comparison manifest")
    validate_baseline_comparison_manifest(manifest, source_dir, report_contract, provenance)
