"""Reconstructible Random Forest and training-mean baseline stage."""

from __future__ import annotations

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
from buffalo_weight.baseline_types import BaselineConfiguration, BaselinePrediction
from buffalo_weight.feature_baselines import RandomForestBaseline
from buffalo_weight.feature_confirmation_manifest import validate_confirmed_feature_package
from buffalo_weight.feature_evaluation import FeatureBaseline, FeatureSample
from buffalo_weight.feature_selection_io import load_feature_samples
from buffalo_weight.feature_selection_artifacts import write_json_artifact
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.snapshot_io import FilesystemSnapshotPublisher, SnapshotPublisher


def run_random_forest_baseline_stage(
    contract: ReportContract, dry_run: bool = False,
    random_forest: FeatureBaseline | None = None,
    provenance: BaselineProvenance | None = None,
    publisher: SnapshotPublisher | None = None,
) -> dict[BaselineConfiguration, str]:
    """Build RF and reference OOF artifacts; for example, dry-run only reports the gate."""
    resolved_provenance = provenance or SystemBaselineProvenance()
    features = validate_confirmed_feature_package(contract)
    statuses = _configuration_statuses(contract, features, resolved_provenance)
    if dry_run:
        return statuses
    if all(status == "reusable" for status in statuses.values()):
        return statuses
    samples = load_feature_samples(contract.inputs_output_dir, features)
    resolved_publisher = publisher or FilesystemSnapshotPublisher()
    return _rebuild_obsolete_configurations(
        contract, samples, features, statuses, random_forest,
        resolved_provenance, resolved_publisher,
    )


def _rebuild_obsolete_configurations(
    contract: ReportContract, samples: list[FeatureSample], features: tuple[str, ...],
    statuses: dict[BaselineConfiguration, str], random_forest: FeatureBaseline | None,
    provenance: BaselineProvenance, publisher: SnapshotPublisher,
) -> dict[BaselineConfiguration, str]:
    if statuses["random_forest_baseline"] != "reusable":
        candidate = evaluate_random_forest_oof(
            samples, features, random_forest or RandomForestBaseline()
        )
        _publish_configuration(contract, "random_forest_baseline", candidate, features,
                               provenance, publisher)
        statuses["random_forest_baseline"] = "rebuilt"
    if statuses["training_mean_reference"] != "reusable":
        reference = evaluate_training_mean_reference(samples)
        _publish_configuration(contract, "training_mean_reference", reference, features,
                               provenance, publisher)
        statuses["training_mean_reference"] = "rebuilt"
    return statuses


def _configuration_statuses(
    contract: ReportContract, features: tuple[str, ...], provenance: BaselineProvenance,
) -> dict[BaselineConfiguration, str]:
    return {
        "random_forest_baseline": baseline_configuration_status(
            contract, "random_forest_baseline", "candidate", features, provenance,
        ),
        "training_mean_reference": baseline_configuration_status(
            contract, "training_mean_reference", "reference", features, provenance,
        ),
    }


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
