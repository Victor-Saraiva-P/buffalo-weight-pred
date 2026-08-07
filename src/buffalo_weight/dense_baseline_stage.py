"""Atomic orchestration for the dense baseline configuration."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from buffalo_weight.artifact_provenance import training_lock
from buffalo_weight.dense_baseline_artifacts import (
    validate_dense_baseline_artifacts,
    write_dense_baseline_artifacts,
)
from buffalo_weight.dense_baseline_evaluation import (
    DenseBaselineEvaluation,
    DenseBaselineRunner,
    DenseFoldAudit,
    ScientificDenseBaselineRunner,
)
from buffalo_weight.dense_baseline_manifest import (
    complete_dense_baseline_manifest,
    dense_baseline_identity,
    dense_baseline_output_dir,
    dense_baseline_status,
    validate_dense_baseline_manifest,
)
from buffalo_weight.dense_baseline_provenance import (
    DenseBaselineProvenance,
    SystemDenseBaselineProvenance,
)
from buffalo_weight.environment_contract import RuntimeProbe
from buffalo_weight.feature_confirmation import (
    baselines_gate_status,
    require_baselines_gate,
)
from buffalo_weight.feature_confirmation_manifest import (
    validate_confirmed_feature_package,
)
from buffalo_weight.feature_evaluation import FeatureSample
from buffalo_weight.feature_selection_io import load_feature_samples
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.snapshot_io import (
    FilesystemSnapshotPublisher,
    SnapshotPublisher,
    clean_snapshot_stage,
)
from buffalo_weight.system_setup import (
    default_runtime_probe,
    require_official_neural_runtime,
)


@dataclass(frozen=True)
class DenseBaselineDependencies:
    """Inject stage boundaries; for example, CLI tests avoid CUDA and Git discovery."""

    runner: DenseBaselineRunner | None = None
    provenance: DenseBaselineProvenance | None = None
    runtime_probe: RuntimeProbe | None = None
    adapter: DenseTrainingAdapter | None = None


def run_dense_baseline_stage(
    contract: ReportContract, dry_run: bool = False,
    publisher: SnapshotPublisher | None = None,
    dependencies: DenseBaselineDependencies | None = None,
) -> str:
    """Run the dense configuration; for example, dry-run reports cache state without CUDA."""
    gate_result = _dense_gate_result(contract, dry_run)
    if gate_result is not None:
        return gate_result
    selected_features = validate_confirmed_feature_package(contract)
    resolved = dependencies or DenseBaselineDependencies()
    resolved_runtime = resolved.runtime_probe or default_runtime_probe()
    resolved_provenance = resolved.provenance or SystemDenseBaselineProvenance(resolved_runtime)
    status = dense_baseline_status(contract, selected_features, resolved_provenance)
    if dry_run or status == "reusable":
        return f"released; dense: {status}"
    return _rebuild_dense_baseline(
        contract, selected_features, resolved, publisher, resolved_provenance, resolved_runtime,
    )


def _dense_gate_result(contract: ReportContract, dry_run: bool) -> str | None:
    gate = cast(str, baselines_gate_status(contract))
    if gate != "released":
        if dry_run:
            return gate
        require_baselines_gate(contract)
    return None


def _rebuild_dense_baseline(
    contract: ReportContract, selected_features: tuple[str, ...],
    dependencies: DenseBaselineDependencies, publisher: SnapshotPublisher | None,
    provenance: DenseBaselineProvenance, runtime_probe: RuntimeProbe,
) -> str:
    with training_lock(Path(contract.artifacts_root) / "baselines"):
        status = dense_baseline_status(contract, selected_features, provenance)
        if status == "reusable":
            return "released; dense: reusable"
        if status == "obsolete":
            clean_snapshot_stage(dense_baseline_output_dir(contract))
        require_official_neural_runtime(False, runtime_probe)
        runner = dependencies.runner or ScientificDenseBaselineRunner()
        resolved_publisher = publisher or FilesystemSnapshotPublisher()
        _build_snapshot(contract, selected_features, runner, resolved_publisher, provenance)
    return "released; dense: rebuilt"


def _build_snapshot(
    contract: ReportContract, selected_features: tuple[str, ...], runner: DenseBaselineRunner,
    publisher: SnapshotPublisher, provenance: DenseBaselineProvenance,
) -> None:
    destination = dense_baseline_output_dir(contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dense-", dir=destination.parent))
    identity = dense_baseline_identity(contract, selected_features, provenance)
    try:
        _write_snapshot(temporary, contract, selected_features, runner, provenance, identity)
        publisher.publish(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_snapshot(
    temporary: Path, contract: ReportContract, selected_features: tuple[str, ...],
    runner: DenseBaselineRunner, provenance: DenseBaselineProvenance,
    identity: dict[str, object],
) -> None:
    samples = load_feature_samples(contract.inputs_output_dir, selected_features)
    evaluation = runner.evaluate(samples, selected_features)
    _validate_evaluation(evaluation, samples, contract.inputs.fold_count)
    write_dense_baseline_artifacts(temporary, evaluation)
    epochs = {audit.fold: audit.selected_epochs for audit in evaluation.fold_audits}
    validate_dense_baseline_artifacts(temporary, samples, contract.inputs.fold_count, epochs)
    _require_stable_identity(contract, selected_features, provenance, identity)
    manifest = complete_dense_baseline_manifest(
        contract, temporary, identity, evaluation, provenance.execution_environment(),
        provenance.repository_commit(),
    )
    validate_dense_baseline_manifest(manifest, temporary, contract)
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _validate_evaluation(
    evaluation: DenseBaselineEvaluation, samples: list[FeatureSample], fold_count: int,
) -> None:
    expected_by_name = {sample.file_name: sample for sample in samples}
    prediction_names = [prediction.file_name for prediction in evaluation.predictions]
    if (len(prediction_names) != len(expected_by_name)
            or set(prediction_names) != set(expected_by_name)):
        raise ValueError(
            f"dense OOF prediction names were {prediction_names!r}; expected one per "
            f"{sorted(expected_by_name)!r}"
        )
    for prediction in evaluation.predictions:
        _validate_prediction_metadata(prediction, expected_by_name[prediction.file_name])
    folds = [audit.fold for audit in evaluation.fold_audits]
    if folds != list(range(1, fold_count + 1)):
        raise ValueError(
            f"dense fold audits were {folds!r}; expected {list(range(1, fold_count + 1))!r}"
        )
    for audit in evaluation.fold_audits:
        _validate_fold_audit(audit, samples)


def _validate_fold_audit(audit: DenseFoldAudit, samples: list[FeatureSample]) -> None:
    selection = set(audit.selection_ids)
    stopping = set(audit.stopping_ids)
    retrain = set(audit.retrain_ids)
    held_out = set(audit.held_out_ids)
    expected_held_out = {sample.file_name for sample in samples if sample.fold == audit.fold}
    expected_retrain = {sample.file_name for sample in samples if sample.fold != audit.fold}
    valid = not selection & stopping and selection | stopping == retrain
    valid = valid and retrain == expected_retrain and held_out == expected_held_out
    valid = valid and 1 <= audit.selected_epochs <= 500
    if not valid:
        actual = (sorted(selection), sorted(stopping), sorted(retrain), sorted(held_out))
        raise ValueError(
            f"dense fold partitions were {actual!r}; expected disjoint inner partitions, "
            f"full retrain {sorted(expected_retrain)!r}, held-out {sorted(expected_held_out)!r}, "
            f"and selected epochs 1..500 but received {audit.selected_epochs!r}"
        )


def _validate_prediction_metadata(
    prediction: object, expected: FeatureSample,
) -> None:
    actual = (
        getattr(prediction, "fold", None), getattr(prediction, "weight_category", None),
        getattr(prediction, "observed_weight_kg", None),
    )
    expected_values = (expected.fold, expected.weight_category, expected.weight_kg)
    if actual != expected_values:
        raise ValueError(
            f"dense prediction metadata for {expected.file_name!r} was {actual!r}; "
            f"expected {expected_values!r}"
        )


def _require_stable_identity(
    contract: ReportContract, selected_features: tuple[str, ...],
    provenance: DenseBaselineProvenance, expected: dict[str, object],
) -> None:
    actual = dense_baseline_identity(contract, selected_features, provenance)
    if actual == expected:
        return
    changed = {key: {"before": expected.get(key), "after": actual.get(key)}
               for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key)}
    raise ValueError(
        f"dense baseline identity changed during training: {changed!r}; expected stable inputs"
    )
