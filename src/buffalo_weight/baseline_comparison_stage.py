"""Atomic stage for the provisional controlled baseline comparison."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from buffalo_weight.baseline_comparison_artifacts import write_baseline_comparison_artifacts
from buffalo_weight.baseline_comparison_inputs import load_comparison_predictions
from buffalo_weight.baseline_comparison_manifest import (
    approach_selection_output_dir,
    baseline_comparison_status,
    comparison_identity,
    complete_comparison_manifest,
    validate_baseline_comparison_manifest,
)
from buffalo_weight.baseline_comparison_provenance import (
    BaselineComparisonProvenance,
    SystemBaselineComparisonProvenance,
)
from buffalo_weight.baseline_provenance import BaselineProvenance
from buffalo_weight.baseline_stage import run_random_forest_baseline_stage
from buffalo_weight.compact_cnn_provenance import CompactCnnProvenance
from buffalo_weight.compact_cnn_stage import run_compact_cnn_stage
from buffalo_weight.dense_baseline_stage import DenseBaselineDependencies, run_dense_baseline_stage
from buffalo_weight.report_provenance import ReportProvenance
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.resnet_baseline_provenance import ResNetBaselineProvenance
from buffalo_weight.resnet_baseline_stage import plan_resnet_baseline_stage
from buffalo_weight.snapshot_io import (
    FilesystemSnapshotPublisher,
    SnapshotPublisher,
    clean_snapshot_stage,
)


@dataclass(frozen=True)
class BaselineComparisonUpstreamDependencies:
    """Inject upstream freshness seams; for example, tests use fixed provenance."""

    report_provenance: ReportProvenance | None = None
    baseline_provenance: BaselineProvenance | None = None
    dense_dependencies: DenseBaselineDependencies | None = None
    compact_provenance: CompactCnnProvenance | None = None
    resnet_provenance: ResNetBaselineProvenance | None = None


def run_baseline_comparison_stage(
    contract: ReportContract, dry_run: bool = False,
    publisher: SnapshotPublisher | None = None,
    provenance: BaselineComparisonProvenance | None = None,
    upstream_dependencies: BaselineComparisonUpstreamDependencies | None = None,
) -> str:
    """Build or reuse comparison evidence; for example, dry-run writes nothing."""
    resolved_upstream = upstream_dependencies or BaselineComparisonUpstreamDependencies()
    _require_reusable_upstream(contract, resolved_upstream)
    comparison_provenance = provenance or SystemBaselineComparisonProvenance()
    status = baseline_comparison_status(contract, comparison_provenance)
    if dry_run or status == "reusable":
        return status
    if status == "obsolete":
        clean_snapshot_stage(approach_selection_output_dir(contract))
    _build_snapshot(
        contract, publisher or FilesystemSnapshotPublisher(), comparison_provenance,
    )
    return "rebuilt"


def _require_reusable_upstream(
    contract: ReportContract, dependencies: BaselineComparisonUpstreamDependencies,
) -> None:
    statuses = _upstream_configuration_statuses(contract, dependencies)
    stale = {name: status for name, status in statuses.items() if status != "reusable"}
    if stale:
        raise ValueError(
            f"baseline comparison inputs were {stale!r}; expected all configurations reusable"
        )


def _upstream_configuration_statuses(
    contract: ReportContract, dependencies: BaselineComparisonUpstreamDependencies,
) -> dict[str, str]:
    feature_statuses = run_random_forest_baseline_stage(
        contract, True, provenance=dependencies.baseline_provenance,
        inputs_provenance=dependencies.report_provenance,
    )
    dense_status = run_dense_baseline_stage(
        contract, True, dependencies=dependencies.dense_dependencies,
    ).rsplit(": ", maxsplit=1)[-1]
    compact_status = run_compact_cnn_stage(
        contract, True, provenance=dependencies.compact_provenance,
    )
    resnet_status = plan_resnet_baseline_stage(contract, dependencies.resnet_provenance)
    return {**feature_statuses, "dense": dense_status, "compact_cnn": compact_status,
            "resnet18_pretrained_partial": resnet_status}


def _build_snapshot(
    contract: ReportContract, publisher: SnapshotPublisher,
    provenance: BaselineComparisonProvenance,
) -> None:
    destination = approach_selection_output_dir(contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".approach-selection-", dir=destination.parent))
    identity = comparison_identity(contract, provenance)
    try:
        _write_snapshot(temporary, contract, identity, provenance)
        publisher.publish(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_snapshot(
    output_dir: Path, contract: ReportContract, identity: dict[str, object],
    provenance: BaselineComparisonProvenance,
) -> None:
    predictions = load_comparison_predictions(contract)
    write_baseline_comparison_artifacts(output_dir, predictions)
    current = comparison_identity(contract, provenance)
    if current != identity:
        raise ValueError(
            f"comparison identity changed from {identity!r} to {current!r}; "
            "expected unchanged baseline inputs"
        )
    manifest = complete_comparison_manifest(output_dir, identity, provenance)
    validate_baseline_comparison_manifest(manifest, output_dir, contract, provenance)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
