"""Atomic stage execution for controlled learning curves diagnostic slice.

Reference: GitHub Issue #25.
"""

from __future__ import annotations

import json
from pathlib import Path

from buffalo_weight.baseline_provenance import BaselineProvenance
from buffalo_weight.compact_cnn_adapter import CompactCnnAdapter
from buffalo_weight.compact_cnn_provenance import CompactCnnProvenance
from buffalo_weight.compact_cnn_types import CompactCnnTrainingAdapter
from buffalo_weight.diagnostic_learning_artifacts import write_learning_curves_artifacts
from buffalo_weight.diagnostic_learning_evaluation import (
    DenseBaselineRunnerSeam,
    evaluate_learning_curves,
)
from buffalo_weight.feature_evaluation import FeatureBaseline
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.resnet_baseline_provenance import ResNetBaselineProvenance
from buffalo_weight.resnet_baseline_stage import (
    ResNetBaselineRunner,
    ScientificResNetBaselineRunner,
)
from buffalo_weight.snapshot_io import (
    FilesystemSnapshotPublisher,
    SnapshotPublisher,
    clean_snapshot_stage,
)


def diagnostic_learning_output_dir(contract: ReportContract) -> Path:
    """Locate output directory for controlled learning curves diagnostic slice.

    Example: ``diagnostic_learning_output_dir(contract)`` returns Path under artifacts_root.
    """
    return contract.artifacts_root / "diagnostics" / "learning_curves"


def run_diagnostic_learning_stage(
    contract: ReportContract,
    dry_run: bool = False,
    publisher: SnapshotPublisher | None = None,
    random_forest_baseline: FeatureBaseline | None = None,
    dense_runner: DenseBaselineRunnerSeam | None = None,
    compact_adapter: CompactCnnTrainingAdapter | None = None,
    resnet_runner: ResNetBaselineRunner | None = None,
    baseline_provenance: BaselineProvenance | None = None,
    compact_provenance: CompactCnnProvenance | None = None,
    resnet_provenance: ResNetBaselineProvenance | None = None,
) -> str:
    """Run the controlled learning curves diagnostic stage.

    Example: ``run_diagnostic_learning_stage(contract, dry_run=True)`` returns "reconstructible".
    """
    output_dir = diagnostic_learning_output_dir(contract)
    manifest_path = output_dir / "manifest.json"

    if dry_run:
        return "reusable" if manifest_path.is_file() else "reconstructible"

    resolved_publisher = publisher or FilesystemSnapshotPublisher()
    resolved_compact_adapter = compact_adapter or CompactCnnAdapter()
    resolved_resnet_runner = resnet_runner or ScientificResNetBaselineRunner()
    return _execute_stage_rebuild(
        contract, output_dir, resolved_publisher, random_forest_baseline,
        dense_runner, resolved_compact_adapter, resolved_resnet_runner,
        baseline_provenance, compact_provenance, resnet_provenance,
    )


def _execute_stage_rebuild(
    contract: ReportContract,
    output_dir: Path,
    publisher: SnapshotPublisher,
    rf_baseline: FeatureBaseline | None,
    dense_runner: DenseBaselineRunnerSeam | None,
    compact_adapter: CompactCnnTrainingAdapter | None,
    resnet_runner: ResNetBaselineRunner | None,
    baseline_provenance: BaselineProvenance | None,
    compact_provenance: CompactCnnProvenance | None,
    resnet_provenance: ResNetBaselineProvenance | None,
) -> str:
    clean_snapshot_stage(output_dir)

    slice_data = evaluate_learning_curves(
        contract,
        random_forest_baseline=rf_baseline,
        dense_runner=dense_runner,
        compact_adapter=compact_adapter,
        resnet_runner=resnet_runner,
        baseline_provenance=baseline_provenance,
        compact_provenance=compact_provenance,
        resnet_provenance=resnet_provenance,
    )

    write_learning_curves_artifacts(output_dir, slice_data)
    _write_stage_manifest(output_dir, slice_data)
    return "rebuilt"


def _write_stage_manifest(output_dir: Path, slice_data: object) -> None:
    point_records = getattr(slice_data, "point_records", ())
    summary_records = getattr(slice_data, "summary_records", ())
    manifest = {
        "status": "complete",
        "stage": "diagnostics_learning",
        "point_count": len(point_records),
        "summary_count": len(summary_records),
        "configurations": sorted(list({getattr(p, "configuration") for p in point_records})),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
