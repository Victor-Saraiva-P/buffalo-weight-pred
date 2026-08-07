"""Atomic publication for confirmed expanded diagnostics evidence package.

Reference: GitHub Issue #27.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from buffalo_weight.diagnostic_confirmation_manifest import (
    build_confirmed_diagnostic_manifest,
    validate_confirmed_diagnostic_manifest,
)
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.snapshot_io import FilesystemSnapshotPublisher, clean_snapshot_stage

SOURCE_DESCRIPTIVE_MANIFEST_NAME = "source_descriptive_manifest.json"
SOURCE_LEARNING_MANIFEST_NAME = "source_learning_manifest.json"
SOURCE_SENSITIVITY_MANIFEST_NAME = "source_sensitivity_manifest.json"


def publish_confirmed_diagnostic_package(
    report_contract: ReportContract,
    human_contract_path: Path,
    reviewed_report_path: Path,
    human_contract: dict[str, object],
    commit: str = "unknown",
) -> None:
    """Publish confirmed diagnostic evidence atomically.

    Example: ``publish_confirmed_diagnostic_package(contract, path_c, path_r, human)``.
    """
    publisher = FilesystemSnapshotPublisher()
    destination = report_contract.confirmed_diagnostics_dir
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".diagnostics-", dir=destination.parent))
    try:
        _write_atomic_package(
            temporary, report_contract, human_contract_path, reviewed_report_path,
            human_contract, commit,
        )
        clean_snapshot_stage(destination)
        publisher.publish(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_atomic_package(
    temporary: Path,
    report_contract: ReportContract,
    human_contract_path: Path,
    reviewed_report_path: Path,
    human_contract: dict[str, object],
    commit: str,
) -> None:
    (temporary / "diagnostics_contract.json").write_text(
        human_contract_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (temporary / "expanded_diagnostics_report.md").write_text(
        reviewed_report_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _copy_source_stage_artifacts(report_contract, temporary)
    manifest = build_confirmed_diagnostic_manifest(temporary, report_contract, human_contract, commit)
    validate_confirmed_diagnostic_manifest(manifest, temporary, human_contract, report_contract)
    (temporary / "manifest.json").write_text(_json_text(manifest), encoding="utf-8")


def _copy_source_stage_artifacts(report_contract: ReportContract, temporary: Path) -> None:
    root = report_contract.artifacts_root / "diagnostics"
    _copy_stage_dir(root / "descriptive", temporary, SOURCE_DESCRIPTIVE_MANIFEST_NAME)
    _copy_stage_dir(root / "learning_curves", temporary, SOURCE_LEARNING_MANIFEST_NAME)
    _copy_stage_dir(root / "sensitivity", temporary, SOURCE_SENSITIVITY_MANIFEST_NAME)


def _copy_stage_dir(source_dir: Path, temporary: Path, manifest_alias: str) -> None:
    if not source_dir.is_dir():
        raise ValueError(f"source diagnostic directory was missing at {source_dir}")
    for item in source_dir.iterdir():
        if item.is_file():
            if item.name == "manifest.json":
                shutil.copy2(item, temporary / manifest_alias)
            elif not (temporary / item.name).exists():
                shutil.copy2(item, temporary / item.name)


def _json_text(value: dict[str, object]) -> str:
    serialized = json.dumps(value, indent=2, sort_keys=True)
    return f"{serialized}\n"
