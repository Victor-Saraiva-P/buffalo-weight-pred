"""Atomic filesystem publication of confirmed feature evidence."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from buffalo_weight.feature_calculators import APPROVED_FEATURES
from buffalo_weight.feature_confirmation_contract import validate_confirmed_feature_contract
from buffalo_weight.feature_confirmation_manifest import (
    CONFIRMED_OUTPUT_FILES,
    SOURCE_MANIFEST_NAME,
    build_confirmed_manifest,
    validate_confirmed_manifest,
)
from buffalo_weight.feature_selection_artifacts import write_json_artifact
from buffalo_weight.feature_selection_manifest import feature_selection_output_dir
from buffalo_weight.feature_selection_validation import validate_feature_selection_evidence_files
from buffalo_weight.reproduction_config import ReportContract

HUMAN_REVIEW_FILES = {"shared_feature_contract.json", "feature_selection_report.md"}


def publish_confirmed_feature_package(
    report_contract: ReportContract, human_contract_path: Path, reviewed_report_path: Path,
    human_contract: dict[str, object],
) -> None:
    """Publish one v1 snapshot; for example, an existing revision is never overwritten."""
    destination = report_contract.confirmed_feature_selection_dir
    if destination.exists() or destination.is_symlink():
        raise ValueError(
            f"confirmed feature destination was {destination}; expected absent v1 destination"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".feature-confirmation-", dir=destination.parent))
    try:
        _write_confirmed_package(
            temporary, report_contract, human_contract_path, reviewed_report_path, human_contract,
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_confirmed_package(
    output_dir: Path, report_contract: ReportContract, human_contract_path: Path,
    reviewed_report_path: Path, human_contract: dict[str, object],
) -> None:
    source_dir = feature_selection_output_dir(report_contract)
    for name in CONFIRMED_OUTPUT_FILES:
        if name not in HUMAN_REVIEW_FILES | {SOURCE_MANIFEST_NAME}:
            shutil.copy2(source_dir / name, output_dir / name)
    shutil.copy2(source_dir / "manifest.json", output_dir / SOURCE_MANIFEST_NAME)
    shutil.copy2(reviewed_report_path, output_dir / "feature_selection_report.md")
    shutil.copy2(human_contract_path, output_dir / "shared_feature_contract.json")
    _validate_copied_evidence(output_dir, human_contract)
    manifest = build_confirmed_manifest(output_dir, report_contract, human_contract)
    validate_confirmed_manifest(manifest, output_dir, human_contract, report_contract)
    write_json_artifact(output_dir / "manifest.json", manifest)


def _validate_copied_evidence(
    output_dir: Path, human_contract: dict[str, object],
) -> None:
    validate_confirmed_feature_contract(
        human_contract, output_dir / "feature_selection_report.md", APPROVED_FEATURES,
    )
    validate_feature_selection_evidence_files(output_dir, APPROVED_FEATURES)
