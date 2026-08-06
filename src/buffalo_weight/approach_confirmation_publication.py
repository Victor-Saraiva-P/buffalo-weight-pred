"""Atomic publication for confirmed approach selection evidence."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from buffalo_weight.approach_confirmation_manifest import (
    SOURCE_MANIFEST_NAME,
    build_confirmed_approach_manifest,
    validate_confirmed_approach_manifest,
)
from buffalo_weight.baseline_comparison_manifest import approach_selection_output_dir
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.snapshot_io import FilesystemSnapshotPublisher, clean_snapshot_stage


def publish_confirmed_approach_package(
    report_contract: ReportContract, human_contract_path: Path,
    reviewed_report_path: Path, human_contract: dict[str, object],
) -> None:
    """Publish confirmed evidence; for example, manifests are written last and verified."""
    publisher = FilesystemSnapshotPublisher()
    destination = report_contract.confirmed_approach_selection_dir
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".approach-", dir=destination.parent))
    try:
        _write_atomic_package(
            temporary, report_contract, human_contract_path, reviewed_report_path,
            human_contract,
        )
        clean_snapshot_stage(destination)
        publisher.publish(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_atomic_package(
    temporary: Path, report_contract: ReportContract, human_contract_path: Path,
    reviewed_report_path: Path, human_contract: dict[str, object],
) -> None:
    source_dir = approach_selection_output_dir(report_contract)
    (temporary / "selected_approach.json").write_text(human_contract_path.read_text())
    (temporary / "approach_selection_report.md").write_text(reviewed_report_path.read_text())
    _copy_source_files(source_dir, temporary)
    manifest = build_confirmed_approach_manifest(temporary, report_contract, human_contract)
    validate_confirmed_approach_manifest(manifest, temporary, human_contract, report_contract)
    (temporary / "manifest.json").write_text(_json_text(manifest))


def _copy_source_files(source_dir: Path, temporary: Path) -> None:
    metrics_source = source_dir / "baseline_metrics.csv"
    manifest_source = source_dir / "manifest.json"
    shutil.copy2(metrics_source, temporary / "baseline_metrics.csv")
    shutil.copy2(manifest_source, temporary / SOURCE_MANIFEST_NAME)


def _json_text(value: dict[str, object]) -> str:
    serialized = json.dumps(value, indent=2, sort_keys=True)
    terminated = f"{serialized}\n"
    return terminated
