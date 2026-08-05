"""Atomic feature-selection stage for the report reproduction pipeline."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Protocol

from buffalo_weight.feature_selection_artifacts import (
    write_feature_selection_artifacts,
    write_json_artifact,
)
from buffalo_weight.feature_baselines import DenseFeatureBaseline, RandomForestBaseline
from buffalo_weight.feature_evaluation import (
    FeatureBaseline,
    FeatureEvidence,
    FeatureSample,
    RemovalGroup,
    evaluate_feature_evidence,
)
from buffalo_weight.feature_selection_contract import PERMUTATION_COUNT, REMOVAL_GROUPS
from buffalo_weight.feature_selection_io import load_feature_samples
from buffalo_weight.feature_selection_manifest import (
    OFFICIAL_EXECUTION,
    complete_feature_selection_manifest,
    feature_selection_identity,
    feature_selection_output_dir,
    feature_selection_status,
    validate_feature_selection_manifest,
)
from buffalo_weight.feature_selection_validation import (
    validate_feature_evidence,
    validate_feature_selection_artifacts,
)
from buffalo_weight.inputs_manifest import stage_status as inputs_stage_status
from buffalo_weight.feature_selection_provenance import (
    FeatureSelectionProvenance,
    SystemFeatureSelectionProvenance,
)
from buffalo_weight.report_provenance import ReportProvenance, SystemReportProvenance
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.snapshot_io import FilesystemSnapshotPublisher, SnapshotPublisher
from buffalo_weight.feature_calculators import APPROVED_FEATURES


class FeatureEvidenceRunner(Protocol):
    """Expensive evaluation seam; for example, acceptance tests inject a named fake."""

    def evaluate(
        self, samples: list[FeatureSample], feature_names: tuple[str, ...],
        removal_groups: tuple[RemovalGroup, ...], permutation_count: int, split_seed: int,
    ) -> list[FeatureEvidence]:
        """Evaluate all experiments; for example, a fake returns complete deterministic rows."""
        ...

    def execution_metadata(self) -> dict[str, object]:
        """Describe execution.

        Example: promotion requires the official CPU/CUDA path.
        """
        ...


class ScientificFeatureEvidenceRunner:
    """Run frozen RF and CUDA dense baselines; for example, production CLI uses this runner."""

    def __init__(self, baselines: tuple[FeatureBaseline, ...] | None = None) -> None:
        """Inject model seams; for example, tests avoid scientific training with fakes."""
        injected_baselines = baselines
        self._baselines = injected_baselines

    def evaluate(
        self, samples: list[FeatureSample], feature_names: tuple[str, ...],
        removal_groups: tuple[RemovalGroup, ...], permutation_count: int, split_seed: int,
    ) -> list[FeatureEvidence]:
        """Produce comparative evidence; for example, permutations reuse each full model."""
        baselines = self._baselines or (RandomForestBaseline(), DenseFeatureBaseline())
        return evaluate_feature_evidence(samples, feature_names, removal_groups, baselines,
                                         permutation_count, split_seed)

    def execution_metadata(self) -> dict[str, object]:
        """Describe execution; for example, injected baselines cannot attest official work."""
        if self._baselines is not None:
            return {"random_forest_device": "injected", "dense_device": "injected",
                    "official": False}
        return OFFICIAL_EXECUTION.copy()


def run_feature_selection_stage(
    contract: ReportContract, dry_run: bool = False,
    runner: FeatureEvidenceRunner | None = None,
    publisher: SnapshotPublisher | None = None,
    inputs_provenance: ReportProvenance | None = None,
    selection_provenance: FeatureSelectionProvenance | None = None,
) -> str:
    """Run selection evidence; for example, dry-run reports status without initializing CUDA."""
    resolved_inputs = inputs_provenance or SystemReportProvenance()
    resolved_selection = selection_provenance or SystemFeatureSelectionProvenance()
    blocked_status = _input_gate_status(contract, dry_run, resolved_inputs)
    if blocked_status is not None:
        return blocked_status
    status = feature_selection_status(contract, resolved_selection)
    if dry_run or status == "reusable":
        return status
    _build_atomic_snapshot(contract, runner or ScientificFeatureEvidenceRunner(),
                           publisher or FilesystemSnapshotPublisher(), resolved_selection)
    return "rebuilt"


def _input_gate_status(
    contract: ReportContract, dry_run: bool, provenance: ReportProvenance,
) -> str | None:
    input_status = inputs_stage_status(contract, provenance)
    if input_status == "reusable":
        return None
    if dry_run:
        return "blocked"
    raise ValueError(
        f"inputs stage status was {input_status!r}; expected 'reusable' before feature-selection"
    )


def _build_atomic_snapshot(
    contract: ReportContract, runner: FeatureEvidenceRunner,
    publisher: SnapshotPublisher, provenance: FeatureSelectionProvenance,
) -> None:
    contract.artifacts_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".feature-selection-", dir=contract.artifacts_root))
    identity = feature_selection_identity(contract, provenance)
    try:
        _write_snapshot(temporary, contract, runner, provenance, identity)
        publisher.publish(temporary, feature_selection_output_dir(contract))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_snapshot(
    output_dir: Path, contract: ReportContract, runner: FeatureEvidenceRunner,
    provenance: FeatureSelectionProvenance, identity: dict[str, object],
) -> None:
    features, samples, evidence = _evaluate_stage(contract, runner)
    write_feature_selection_artifacts(output_dir, samples, features, evidence)
    validate_feature_selection_artifacts(output_dir, features)
    _require_unchanged_identity(contract, provenance, identity)
    _write_stage_manifest(output_dir, contract, identity, runner.execution_metadata())


def _evaluate_stage(
    contract: ReportContract, runner: FeatureEvidenceRunner,
) -> tuple[tuple[str, ...], list[FeatureSample], list[FeatureEvidence]]:
    features = tuple(APPROVED_FEATURES)
    samples = load_feature_samples(contract.inputs_output_dir, features)
    evidence = runner.evaluate(samples, features, REMOVAL_GROUPS, PERMUTATION_COUNT,
                               contract.inputs.fold_seed)
    validate_feature_evidence(evidence, samples, features, REMOVAL_GROUPS, PERMUTATION_COUNT,
                              contract.inputs.fold_seed)
    return features, samples, evidence


def _require_unchanged_identity(
    contract: ReportContract, provenance: FeatureSelectionProvenance,
    identity: dict[str, object],
) -> None:
    current = feature_selection_identity(contract, provenance)
    if current != identity:
        raise ValueError(
            f"selection identity changed from {identity!r} to {current!r}; "
            "expected an unchanged snapshot"
        )


def _write_stage_manifest(
    output_dir: Path, contract: ReportContract, identity: dict[str, object],
    execution: dict[str, object],
) -> None:
    source_commit = identity.get("source_commit")
    if not isinstance(source_commit, str):
        raise ValueError(
            f"selection source commit was {source_commit!r}; expected a Git SHA string"
        )
    manifest = complete_feature_selection_manifest(
        contract, output_dir, identity, source_commit, execution
    )
    validate_feature_selection_manifest(manifest, output_dir)
    write_json_artifact(output_dir / "manifest.json", manifest)
