"""Atomic stage execution for controlled sensitivity diagnostic slice.

Reference: GitHub Issue #26.
"""

from __future__ import annotations

import json
from pathlib import Path

from buffalo_weight.diagnostic_sensitivity_artifacts import write_sensitivity_artifacts
from buffalo_weight.diagnostic_sensitivity_evaluation import (
    FilesystemMaskLoader,
    SensitivityMaskLoader,
    evaluate_sensitivity,
)
from buffalo_weight.diagnostic_sensitivity_types import SensitivitySlice
from buffalo_weight.feature_evaluation import FeatureBaseline
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.snapshot_io import (
    FilesystemSnapshotPublisher,
    SnapshotPublisher,
    clean_snapshot_stage,
)


def diagnostic_sensitivity_output_dir(contract: ReportContract) -> Path:
    """Locate output directory for controlled sensitivity diagnostic slice.

    Example: ``diagnostic_sensitivity_output_dir(contract)`` returns Path under artifacts_root.
    """
    return contract.artifacts_root / "diagnostics" / "sensitivity"


def run_diagnostic_sensitivity_stage(
    contract: ReportContract,
    dry_run: bool = False,
    publisher: SnapshotPublisher | None = None,
    mask_loader: SensitivityMaskLoader | None = None,
    random_forest_baseline: FeatureBaseline | None = None,
) -> str:
    """Run the controlled sensitivity diagnostic stage.

    Example: ``run_diagnostic_sensitivity_stage(contract, dry_run=True)`` returns "reconstructible".
    """
    output_dir = diagnostic_sensitivity_output_dir(contract)
    manifest_path = output_dir / "manifest.json"

    if dry_run:
        return "reusable" if manifest_path.is_file() else "reconstructible"

    resolved_publisher = publisher or FilesystemSnapshotPublisher()
    return _execute_stage_rebuild(
        contract, output_dir, resolved_publisher, mask_loader, random_forest_baseline,
    )


def _execute_stage_rebuild(
    contract: ReportContract, output_dir: Path, publisher: SnapshotPublisher,
    mask_loader: SensitivityMaskLoader | None, rf_baseline: FeatureBaseline | None,
) -> str:
    clean_snapshot_stage(output_dir)
    target_slice = evaluate_sensitivity(contract, mask_loader=mask_loader, random_forest_baseline=rf_baseline)
    loader = mask_loader or FilesystemMaskLoader(contract.inputs.masks_dir)
    demo_masks = _load_demo_masks(target_slice, loader)
    write_sensitivity_artifacts(output_dir, target_slice, demo_masks)
    _write_stage_manifest(output_dir, target_slice)
    return "rebuilt"


def _load_demo_masks(
    slice_data: SensitivitySlice,
    loader: SensitivityMaskLoader,
) -> dict[str, object]:
    """Load masks for the demo PNG — only the first eligible mask."""
    import numpy as np
    for e in slice_data.eligibilities:
        if e.status == "eligible":
            return {e.file_name: loader.load_mask(e.file_name)}
    return {}


def _write_stage_manifest(output_dir: Path, slice_data: SensitivitySlice) -> None:
    eligible_count = sum(1 for e in slice_data.eligibilities if e.status == "eligible")
    rejected_count = sum(1 for e in slice_data.eligibilities if e.status == "rejected")
    manifest = {
        "configurations": sorted({r.configuration for r in slice_data.records}),
        "eligible_mask_count": eligible_count,
        "rejected_mask_count": rejected_count,
        "stage": "diagnostics_sensitivity",
        "status": "complete",
        "total_mask_count": len(slice_data.eligibilities),
        "total_perturbation_records": len(slice_data.records),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
