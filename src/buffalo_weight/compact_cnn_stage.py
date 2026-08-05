"""Atomic orchestration for the compact CNN baseline configuration."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from buffalo_weight.artifact_provenance import training_lock
from buffalo_weight.compact_cnn_adapter import CompactCnnAdapter
from buffalo_weight.compact_cnn_types import (
    COMPACT_CNN_RECIPE,
    CompactCnnRecipe,
    CompactCnnRunStatus,
    CompactCnnTrainingAdapter,
)
from buffalo_weight.compact_cnn_artifacts import write_compact_cnn_artifacts
from buffalo_weight.compact_cnn_manifest import (
    build_compact_cnn_manifest,
    compact_cnn_output_dir,
    compact_cnn_status,
    validate_compact_cnn_manifest,
    write_manifest_last,
)
from buffalo_weight.compact_cnn_evaluation import (
    CompactCnnEvaluation,
    evaluate_compact_cnn,
    load_compact_cnn_samples,
)
from buffalo_weight.compact_cnn_provenance import (
    CompactCnnProvenance,
    SystemCompactCnnProvenance,
)
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.snapshot_io import (
    FilesystemSnapshotPublisher,
    SnapshotPublisher,
    clean_snapshot_stage,
)


def run_compact_cnn_stage(
    contract: ReportContract, dry_run: bool = False,
    adapter: CompactCnnTrainingAdapter | None = None,
    provenance: CompactCnnProvenance | None = None,
    publisher: SnapshotPublisher | None = None,
    recipe: CompactCnnRecipe = COMPACT_CNN_RECIPE,
) -> CompactCnnRunStatus:
    """Run one baseline; for example, dry-run classifies without initializing CUDA."""
    resolved_provenance = provenance or SystemCompactCnnProvenance()
    status = compact_cnn_status(contract, recipe, resolved_provenance)
    if dry_run or status == "reusable":
        return status
    lock_dir = contract.artifacts_root / ".locks" / "compact_cnn"
    with training_lock(lock_dir):
        return _run_locked_rebuild(
            contract, adapter, resolved_provenance, publisher, recipe,
        )


def _run_locked_rebuild(
    contract: ReportContract, adapter: CompactCnnTrainingAdapter | None,
    provenance: CompactCnnProvenance, publisher: SnapshotPublisher | None,
    recipe: CompactCnnRecipe,
) -> CompactCnnRunStatus:
    status = compact_cnn_status(contract, recipe, provenance)
    if status == "reusable":
        return status
    if status == "obsolete":
        clean_snapshot_stage(compact_cnn_output_dir(contract))
    resolved_adapter = adapter or CompactCnnAdapter()
    samples = load_compact_cnn_samples(
        contract.inputs_output_dir / "canonical_split.csv", contract.inputs.masks_dir,
    )
    evaluation = evaluate_compact_cnn(samples, resolved_adapter, recipe)
    resolved_publisher = publisher or FilesystemSnapshotPublisher()
    _publish_evaluation(contract, evaluation, recipe, provenance, resolved_publisher)
    return "rebuilt"


def _publish_evaluation(
    contract: ReportContract, evaluation: CompactCnnEvaluation, recipe: CompactCnnRecipe,
    provenance: CompactCnnProvenance, publisher: SnapshotPublisher,
) -> None:
    parent = contract.artifacts_root / "baselines"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".compact-cnn-", dir=parent))
    try:
        _write_snapshot(temporary, contract, evaluation, recipe, provenance)
        publisher.publish(temporary, compact_cnn_output_dir(contract))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_snapshot(
    temporary: Path, contract: ReportContract, evaluation: CompactCnnEvaluation,
    recipe: CompactCnnRecipe, provenance: CompactCnnProvenance,
) -> None:
    write_compact_cnn_artifacts(temporary, evaluation)
    manifest = build_compact_cnn_manifest(temporary, contract, recipe, provenance)
    validate_compact_cnn_manifest(manifest, temporary, contract, recipe, provenance)
    write_manifest_last(temporary / "manifest.json", manifest)
