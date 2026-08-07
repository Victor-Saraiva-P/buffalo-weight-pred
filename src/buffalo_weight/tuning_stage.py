"""Atomic stage execution for pre-registered configuration tuning."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from buffalo_weight.artifact_provenance import training_lock
from buffalo_weight.baseline_comparison_inputs import load_comparison_predictions
from buffalo_weight.baseline_comparison_metrics import comparison_metric_rows
from buffalo_weight.baseline_comparison_types import ComparisonPrediction
from buffalo_weight.compact_cnn_types import CompactCnnTrainingAdapter
from buffalo_weight.dense_feature_adapter import DenseFeatureAdapter
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.resnet_baseline_stage import ResNetBaselineRunner
from buffalo_weight.snapshot_io import (
    FilesystemSnapshotPublisher,
    SnapshotPublisher,
    clean_snapshot_stage,
)
from buffalo_weight.tuning_artifacts import write_tuning_artifacts
from buffalo_weight.tuning_evaluation import evaluate_tuning_variations
from buffalo_weight.tuning_inputs import validate_tuning_gate_and_contract
from buffalo_weight.tuning_manifest import (
    build_tuning_manifest,
    tuning_output_dir,
    tuning_stage_status,
    validate_tuning_manifest,
)
from buffalo_weight.tuning_provenance import SystemTuningProvenance, TuningProvenance


def run_tuning_stage(
    contract: ReportContract, dry_run: bool = False,
    provenance: TuningProvenance | None = None,
    publisher: SnapshotPublisher | None = None,
    dense_adapter: DenseFeatureAdapter | None = None,
    compact_cnn_adapter: CompactCnnTrainingAdapter | None = None,
    resnet_runner: ResNetBaselineRunner | None = None,
) -> str:
    """Run configuration tuning stage; for example, ``run_tuning_stage(contract, dry_run=True)``."""
    approach, baseline_config, budget, frozen_features, variations = validate_tuning_gate_and_contract(contract)
    resolved_provenance = provenance or SystemTuningProvenance()
    status = tuning_stage_status(contract, resolved_provenance)
    if dry_run or status == "reusable":
        return status
    if budget <= 0 or not variations:
        _write_baseline_maintained_manifest(contract, approach, baseline_config, budget, resolved_provenance)
        return "released; baseline_maintained"
    if status == "obsolete":
        clean_snapshot_stage(tuning_output_dir(contract))
    lock_dir = contract.artifacts_root / ".locks" / "tuning"
    with training_lock(lock_dir):
        return _run_locked_tuning(
            contract, approach, baseline_config, budget, frozen_features,
            variations, resolved_provenance, publisher or FilesystemSnapshotPublisher(),
            dense_adapter, compact_cnn_adapter, resnet_runner,
        )


def _run_locked_tuning(
    contract: ReportContract, approach: str, baseline_config: str, budget: int,
    frozen_features: tuple[str, ...] | None, variations: tuple,
    provenance: TuningProvenance, publisher: SnapshotPublisher,
    dense_adapter: DenseFeatureAdapter | None,
    compact_cnn_adapter: CompactCnnTrainingAdapter | None,
    resnet_runner: ResNetBaselineRunner | None,
) -> str:
    status = tuning_stage_status(contract, provenance)
    if status == "reusable":
        return status
    destination = tuning_output_dir(contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tuning-", dir=destination.parent))
    try:
        _write_snapshot(
            temporary, contract, approach, baseline_config, budget,
            frozen_features, variations, provenance, dense_adapter,
            compact_cnn_adapter, resnet_runner,
        )
        publisher.publish(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "rebuilt"


def _write_snapshot(
    temporary: Path, contract: ReportContract, approach: str, baseline_config: str,
    budget: int, frozen_features: tuple[str, ...] | None, variations: tuple,
    provenance: TuningProvenance, dense_adapter: DenseFeatureAdapter | None,
    compact_cnn_adapter: CompactCnnTrainingAdapter | None,
    resnet_runner: ResNetBaselineRunner | None,
) -> None:
    baseline_preds = [p for p in load_comparison_predictions(contract) if p.configuration == baseline_config]
    tuned_preds, metrics = evaluate_tuning_variations(
        contract, approach, baseline_config, frozen_features, variations,
        dense_adapter, compact_cnn_adapter, resnet_runner,
    )
    all_preds = [*baseline_preds, *tuned_preds]
    baseline_metrics = comparison_metric_rows(baseline_preds)
    all_metrics = [*baseline_metrics, *metrics]
    write_tuning_artifacts(temporary, all_preds, all_metrics, approach, baseline_config, variations)
    manifest = build_tuning_manifest(
        temporary, contract, approach, baseline_config, budget, variations, provenance,
    )
    validate_tuning_manifest(manifest, temporary, contract, provenance)
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_baseline_maintained_manifest(
    contract: ReportContract, approach: str, baseline_config: str,
    budget: int, provenance: TuningProvenance,
) -> None:
    output_dir = tuning_output_dir(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_tuning_manifest(
        output_dir, contract, approach, baseline_config, budget, (), provenance,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
