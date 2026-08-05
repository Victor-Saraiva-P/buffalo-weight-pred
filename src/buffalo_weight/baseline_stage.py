"""Reconstructible Random Forest and training-mean baseline stage."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from buffalo_weight.baseline_artifacts import write_baseline_metrics, write_baseline_predictions
from buffalo_weight.baseline_evaluation import (
    evaluate_random_forest_oof,
    evaluate_training_mean_reference,
)
from buffalo_weight.baseline_manifest import (
    baseline_configuration_status,
    complete_baseline_manifest,
)
from buffalo_weight.baseline_provenance import BaselineProvenance, SystemBaselineProvenance
from buffalo_weight.baseline_types import (
    BaselineConfiguration,
    BaselinePrediction,
    BaselineStatus,
)
from buffalo_weight.feature_baselines import RandomForestBaseline
from buffalo_weight.feature_confirmation_manifest import validate_frozen_feature_contract
from buffalo_weight.feature_evaluation import FeatureBaseline, FeatureSample
from buffalo_weight.feature_selection_io import load_feature_samples
from buffalo_weight.feature_selection_artifacts import write_json_artifact
from buffalo_weight.inputs_manifest import expected_identity as expected_inputs_identity
from buffalo_weight.report_provenance import ReportProvenance, SystemReportProvenance
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.snapshot_io import (
    FilesystemSnapshotPublisher,
    SnapshotPublisher,
    clean_snapshot_stage,
)


def run_random_forest_baseline_stage(
    contract: ReportContract, dry_run: bool = False,
    random_forest: FeatureBaseline | None = None,
    provenance: BaselineProvenance | None = None,
    publisher: SnapshotPublisher | None = None,
    inputs_provenance: ReportProvenance | None = None,
) -> dict[BaselineConfiguration, BaselineStatus]:
    """Build RF and reference OOF artifacts; for example, dry-run only reports the gate."""
    resolved_provenance = provenance or SystemBaselineProvenance()
    features = validate_frozen_feature_contract(contract)
    resolved_inputs = inputs_provenance or SystemReportProvenance()
    if not _inputs_identity_is_current(contract, resolved_inputs):
        return _blocked_or_raise(contract, resolved_inputs, dry_run)
    return _run_current_baseline_inputs(
        contract, features, dry_run, random_forest, resolved_provenance, publisher,
    )


def _run_current_baseline_inputs(
    contract: ReportContract, features: tuple[str, ...], dry_run: bool,
    random_forest: FeatureBaseline | None, provenance: BaselineProvenance,
    publisher: SnapshotPublisher | None,
) -> dict[BaselineConfiguration, BaselineStatus]:
    statuses = _configuration_statuses(contract, features, provenance)
    if dry_run:
        return statuses
    if all(status == "reusable" for status in statuses.values()):
        return statuses
    samples = load_feature_samples(contract.inputs_output_dir, features)
    resolved_publisher = publisher or FilesystemSnapshotPublisher()
    return _rebuild_obsolete_configurations(
        contract, samples, features, statuses, random_forest,
        provenance, resolved_publisher,
    )


def _rebuild_obsolete_configurations(
    contract: ReportContract, samples: list[FeatureSample], features: tuple[str, ...],
    statuses: dict[BaselineConfiguration, BaselineStatus],
    random_forest: FeatureBaseline | None,
    provenance: BaselineProvenance, publisher: SnapshotPublisher,
) -> dict[BaselineConfiguration, BaselineStatus]:
    if statuses["random_forest_baseline"] != "reusable":
        _remove_obsolete(contract, "random_forest_baseline", statuses)
        candidate = evaluate_random_forest_oof(samples, features,
                                               random_forest or RandomForestBaseline())
        _publish_configuration(contract, "random_forest_baseline", candidate,
                               features, provenance, publisher)
        statuses["random_forest_baseline"] = "rebuilt"
    if statuses["training_mean_reference"] != "reusable":
        _remove_obsolete(contract, "training_mean_reference", statuses)
        reference = evaluate_training_mean_reference(samples)
        _publish_configuration(contract, "training_mean_reference", reference, features,
                               provenance, publisher)
        statuses["training_mean_reference"] = "rebuilt"
    return statuses


def _configuration_statuses(
    contract: ReportContract, features: tuple[str, ...], provenance: BaselineProvenance,
) -> dict[BaselineConfiguration, BaselineStatus]:
    return {
        "random_forest_baseline": baseline_configuration_status(
            contract, "random_forest_baseline", "candidate", features, provenance,
        ),
        "training_mean_reference": baseline_configuration_status(
            contract, "training_mean_reference", "reference", features, provenance,
        ),
    }


def _inputs_identity_is_current(
    contract: ReportContract, provenance: ReportProvenance,
) -> bool:
    manifest_path = contract.inputs_output_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        expected = expected_inputs_identity(contract, provenance)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(manifest, dict) and all(
        manifest.get(key) == value for key, value in expected.items()
    )


def _blocked_or_raise(
    contract: ReportContract, provenance: ReportProvenance, dry_run: bool,
) -> dict[BaselineConfiguration, BaselineStatus]:
    if dry_run:
        return {"random_forest_baseline": "blocked", "training_mean_reference": "blocked"}
    manifest_path = contract.inputs_output_dir / "manifest.json"
    actual = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    expected = expected_inputs_identity(contract, provenance)
    raise ValueError(f"inputs identity was {actual!r}; expected current identity {expected!r}")


def _remove_obsolete(
    contract: ReportContract, configuration: BaselineConfiguration,
    statuses: dict[BaselineConfiguration, BaselineStatus],
) -> None:
    if statuses[configuration] != "obsolete":
        return
    output_dir = contract.artifacts_root / "baselines" / configuration
    clean_snapshot_stage(output_dir)


def _publish_configuration(
    contract: ReportContract, configuration: BaselineConfiguration,
    predictions: list[BaselinePrediction], features: tuple[str, ...],
    provenance: BaselineProvenance, publisher: SnapshotPublisher,
) -> None:
    baselines_dir = contract.artifacts_root / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{configuration}-", dir=baselines_dir))
    try:
        write_baseline_predictions(temporary / "predictions.csv", predictions)
        write_baseline_metrics(temporary, predictions)
        role = predictions[0].evaluation_role
        manifest = complete_baseline_manifest(
            contract, temporary, configuration, role, features, provenance,
        )
        write_json_artifact(temporary / "manifest.json", manifest)
        publisher.publish(temporary, baselines_dir / configuration)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
