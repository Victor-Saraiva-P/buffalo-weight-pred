"""Atomic orchestration for the compact CNN baseline configuration."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from buffalo_weight.compact_cnn_adapter import (
    COMPACT_CNN_RECIPE,
    CompactCnnAdapter,
    CompactCnnRecipe,
    CompactCnnTrainingAdapter,
)
from buffalo_weight.compact_cnn_artifacts import (
    build_compact_cnn_manifest,
    compact_cnn_output_dir,
    compact_cnn_status,
    validate_compact_cnn_manifest,
    write_compact_cnn_artifacts,
    write_manifest_last,
)
from buffalo_weight.compact_cnn_evaluation import (
    evaluate_compact_cnn,
    load_compact_cnn_samples,
)
from buffalo_weight.compact_cnn_provenance import (
    CompactCnnProvenance,
    SystemCompactCnnProvenance,
)
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.snapshot_io import FilesystemSnapshotPublisher, SnapshotPublisher


def run_compact_cnn_stage(
    contract: ReportContract, dry_run: bool = False,
    adapter: CompactCnnTrainingAdapter | None = None,
    provenance: CompactCnnProvenance | None = None,
    publisher: SnapshotPublisher | None = None,
    recipe: CompactCnnRecipe = COMPACT_CNN_RECIPE,
) -> str:
    """Run one baseline; for example, dry-run classifies without initializing CUDA."""
    resolved_provenance = provenance or SystemCompactCnnProvenance()
    status = compact_cnn_status(contract, recipe, resolved_provenance)
    if dry_run or status == "reusable":
        return status
    resolved_adapter = adapter or CompactCnnAdapter()
    samples = load_compact_cnn_samples(
        contract.inputs_output_dir / "canonical_split.csv", contract.inputs.masks_dir,
    )
    evaluation = evaluate_compact_cnn(samples, resolved_adapter, recipe)
    resolved_publisher = publisher or FilesystemSnapshotPublisher()
    _publish_evaluation(contract, evaluation, recipe, resolved_provenance, resolved_publisher)
    return "rebuilt"


def _publish_evaluation(
    contract: ReportContract, evaluation: object, recipe: CompactCnnRecipe,
    provenance: CompactCnnProvenance, publisher: SnapshotPublisher,
) -> None:
    from buffalo_weight.compact_cnn_evaluation import CompactCnnEvaluation

    if not isinstance(evaluation, CompactCnnEvaluation):
        raise ValueError(f"compact CNN evaluation was {evaluation!r}; expected typed evidence")
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
    temporary: Path, contract: ReportContract, evaluation: object,
    recipe: CompactCnnRecipe, provenance: CompactCnnProvenance,
) -> None:
    from buffalo_weight.compact_cnn_evaluation import CompactCnnEvaluation

    if not isinstance(evaluation, CompactCnnEvaluation):
        raise ValueError(f"compact CNN evaluation was {evaluation!r}; expected typed evidence")
    write_compact_cnn_artifacts(temporary, evaluation)
    manifest = build_compact_cnn_manifest(temporary, contract, recipe, provenance)
    validate_compact_cnn_manifest(manifest, temporary, contract, recipe, provenance)
    write_manifest_last(temporary / "manifest.json", manifest)
