"""Promotion orchestration and downstream gate for confirmed feature evidence."""

from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Literal

from buffalo_weight.feature_calculators import APPROVED_FEATURES
from buffalo_weight.feature_confirmation_contract import validate_confirmed_feature_contract
from buffalo_weight.feature_confirmation_environment import (
    FeatureConfirmationEnvironment,
    LocalFeatureConfirmationEnvironment,
)
from buffalo_weight.feature_confirmation_manifest import (
    read_json_mapping,
    validate_confirmed_feature_package,
)
from buffalo_weight.feature_confirmation_publication import publish_confirmed_feature_package
from buffalo_weight.feature_selection_manifest import (
    OFFICIAL_EXECUTION,
    feature_selection_output_dir,
    feature_selection_status,
    validate_feature_selection_manifest,
)
from buffalo_weight.feature_selection_provenance import (
    FeatureSelectionProvenance,
    SystemFeatureSelectionProvenance,
)
from buffalo_weight.feature_selection_validation import validate_feature_selection_artifacts
from buffalo_weight.reproduction_config import ReportContract


@dataclass(frozen=True)
class _GateInspection:
    state: Literal["blocked", "released"]
    reason: str | None

    def message(self) -> str:
        """Render user status; for example, blocked gates include their reason."""
        if self.reason is None:
            return self.state
        return f"{self.state}: {self.reason}"


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
    if isinstance(validation, str) or dry_run:
        return validation if isinstance(validation, str) else "released"
    publish_confirmed_feature_package(
        report_contract, human_contract_path, reviewed_report_path, validation,
    )
    return "confirmed"


def baselines_gate_status(report_contract: ReportContract) -> str:
    """Explain the baseline gate.

    Example: tampering returns ``blocked: <reason>``.
    """
    return _inspect_baselines_gate(report_contract).message()


def require_baselines_gate(report_contract: ReportContract) -> None:
    """Fail before baseline work; for example, a missing confirmation raises clearly."""
    inspection = _inspect_baselines_gate(report_contract)
    if inspection.state == "released":
        return
    raise ValueError(
        f"confirmed feature gate was blocked by {inspection.reason}; expected an intact "
        "compatible package at evidence/confirmed/feature_selection/v1"
    )


def _inspect_baselines_gate(report_contract: ReportContract) -> _GateInspection:
    try:
        validate_confirmed_feature_package(report_contract)
    except (OSError, ValueError, TypeError, JSONDecodeError) as error:
        return _GateInspection("blocked", str(error))
    return _GateInspection("released", None)


def _promotion_validation(
    report_contract: ReportContract, human_contract_path: Path, reviewed_report_path: Path,
    dry_run: bool, environment: FeatureConfirmationEnvironment,
    provenance: FeatureSelectionProvenance,
) -> dict[str, object] | str:
    try:
        return _validate_promotion(
            report_contract, human_contract_path, reviewed_report_path,
            environment, provenance,
        )
    except (OSError, ValueError, TypeError, JSONDecodeError) as error:
        if dry_run:
            return f"blocked: {error}"
        raise


def _validate_promotion(
    report_contract: ReportContract, human_contract_path: Path, reviewed_report_path: Path,
    environment: FeatureConfirmationEnvironment, provenance: FeatureSelectionProvenance,
) -> dict[str, object]:
    _require_clean_worktree(environment)
    _validate_source_package(report_contract, provenance)
    human_contract = read_json_mapping(human_contract_path, "human feature contract")
    validate_confirmed_feature_contract(human_contract, reviewed_report_path, APPROVED_FEATURES)
    return human_contract


def _require_clean_worktree(environment: FeatureConfirmationEnvironment) -> None:
    changes = environment.worktree_changes(Path(__file__).parents[2])
    if changes:
        raise ValueError(f"worktree changes were {changes!r}; expected a clean worktree")


def _validate_source_package(
    report_contract: ReportContract, provenance: FeatureSelectionProvenance,
) -> None:
    status = feature_selection_status(report_contract, provenance)
    if status != "reusable":
        raise ValueError(
            f"feature-selection stage status was {status!r}; expected reusable reviewed evidence"
        )
    source_dir = feature_selection_output_dir(report_contract)
    manifest = read_json_mapping(source_dir / "manifest.json", "feature manifest")
    validate_feature_selection_manifest(manifest, source_dir)
    validate_feature_selection_artifacts(source_dir, APPROVED_FEATURES)
    if manifest.get("execution") != OFFICIAL_EXECUTION:
        raise ValueError(
            f"feature execution was {manifest.get('execution')!r}; "
            f"expected official execution {OFFICIAL_EXECUTION!r}"
        )
