"""Atomic feature-selection stage for the report reproduction pipeline."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

from PIL import Image

from buffalo_weight.csv_io import csv_columns
from buffalo_weight.feature_baselines import DenseFeatureBaseline, RandomForestBaseline
from buffalo_weight.feature_evaluation import (
    FeatureEvidence,
    FeatureSample,
    RemovalGroup,
    evaluate_feature_evidence,
)
from buffalo_weight.feature_recommendations import (
    build_removal_recommendations,
    provisional_feature_contract,
)
from buffalo_weight.feature_redundancy import calculate_feature_redundancy
from buffalo_weight.feature_selection_contract import PERMUTATION_COUNT, REMOVAL_GROUPS
from buffalo_weight.feature_selection_io import (
    load_feature_samples,
    write_feature_evidence,
    write_feature_redundancy,
)
from buffalo_weight.feature_selection_manifest import (
    complete_feature_selection_manifest,
    expected_csv_schemas,
    feature_selection_identity,
    feature_selection_output_dir,
    feature_selection_status,
    validate_feature_selection_manifest,
)
from buffalo_weight.feature_selection_plots import save_feature_selection_figures
from buffalo_weight.feature_selection_report import selection_report_markdown
from buffalo_weight.feature_selection_validation import validate_feature_evidence
from buffalo_weight.hashing import sha256_file
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
    ) -> list[FeatureEvidence]: ...


class ScientificFeatureEvidenceRunner:
    """Run frozen RF and CUDA dense baselines; for example, production CLI uses this runner."""

    def evaluate(
        self, samples: list[FeatureSample], feature_names: tuple[str, ...],
        removal_groups: tuple[RemovalGroup, ...], permutation_count: int, split_seed: int,
    ) -> list[FeatureEvidence]:
        """Produce comparative evidence; for example, permutations reuse each full model."""
        baselines = (RandomForestBaseline(), DenseFeatureBaseline())
        return evaluate_feature_evidence(samples, feature_names, removal_groups, baselines,
                                         permutation_count, split_seed)


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
    if inputs_stage_status(contract, resolved_inputs) != "reusable":
        if dry_run:
            return "blocked"
        raise ValueError("inputs stage was not reusable; expected current inputs before feature-selection")
    status = feature_selection_status(contract, resolved_selection)
    if dry_run or status == "reusable":
        return status
    _build_atomic_snapshot(contract, runner or ScientificFeatureEvidenceRunner(),
                           publisher or FilesystemSnapshotPublisher(), resolved_selection)
    return "rebuilt"


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
    features = tuple(APPROVED_FEATURES)
    samples = load_feature_samples(contract.inputs_output_dir, features)
    evidence = runner.evaluate(samples, features, REMOVAL_GROUPS, PERMUTATION_COUNT,
                               contract.inputs.fold_seed)
    validate_feature_evidence(evidence, samples, features, REMOVAL_GROUPS, PERMUTATION_COUNT,
                              contract.inputs.fold_seed)
    _write_public_outputs(output_dir, samples, features, evidence)
    _validate_public_outputs(output_dir)
    if feature_selection_identity(contract, provenance) != identity:
        raise ValueError("selection inputs changed during the stage; expected an unchanged snapshot")
    manifest = complete_feature_selection_manifest(
        contract, output_dir, identity, provenance.repository_commit()
    )
    validate_feature_selection_manifest(manifest, output_dir)
    (output_dir / "manifest.json").write_text(_json_text(manifest))


def _write_public_outputs(
    output_dir: Path, samples: list[FeatureSample], features: tuple[str, ...],
    evidence: list[FeatureEvidence],
) -> None:
    redundancy = calculate_feature_redundancy(samples, features)
    targets = (*features, *(group.name for group in REMOVAL_GROUPS))
    recommendations = build_removal_recommendations(evidence, targets)
    write_feature_redundancy(output_dir / "feature_redundancy.csv", redundancy)
    write_feature_evidence(output_dir / "feature_predictive_evidence.csv", evidence)
    report_path = output_dir / "feature_selection_report.md"
    report_path.write_text(selection_report_markdown(recommendations))
    contract = provisional_feature_contract(features, recommendations, sha256_file(report_path))
    (output_dir / "shared_feature_contract.json").write_text(_json_text(contract))
    save_feature_selection_figures(output_dir, features, redundancy, evidence, recommendations)


def _validate_public_outputs(output_dir: Path) -> None:
    for name, expected in expected_csv_schemas().items():
        actual = csv_columns(output_dir / name)
        if actual != expected:
            raise ValueError(f"selection columns were {actual!r} for {name}; expected {expected!r}")
    contract = json.loads((output_dir / "shared_feature_contract.json").read_text())
    if contract.get("status") != "provisional" or contract.get("selected_features") is not None:
        raise ValueError(f"shared feature contract was {contract!r}; expected an unselected provisional contract")
    _validate_figures(output_dir)


def _validate_figures(output_dir: Path) -> None:
    for name in ("redundancy_heatmap.png", "removal_heatmap.png", "permutation_effects.png"):
        with Image.open(output_dir / name) as figure:
            dpi = float(figure.info.get("dpi", (0.0, 0.0))[0])
            if figure.size != (2400, 1800) or round(dpi) != 300:
                raise ValueError(
                    f"figure shape/DPI was {figure.size}/{dpi} for {name}; expected (2400, 1800)/300"
                )


def _json_text(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"
