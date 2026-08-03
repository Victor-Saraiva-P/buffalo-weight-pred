"""Serialization of the provisional feature-selection artifact package."""

from __future__ import annotations

import json
from pathlib import Path

from buffalo_weight.feature_evaluation import FeatureEvidence, FeatureSample
from buffalo_weight.feature_recommendations import (
    build_removal_recommendations,
    provisional_feature_contract,
)
from buffalo_weight.feature_redundancy import calculate_feature_redundancy
from buffalo_weight.feature_selection_contract import REMOVAL_GROUPS
from buffalo_weight.feature_selection_io import write_feature_evidence, write_feature_redundancy
from buffalo_weight.feature_selection_plots import save_feature_selection_figures
from buffalo_weight.feature_selection_report import selection_report_markdown
from buffalo_weight.hashing import sha256_file


def write_feature_selection_artifacts(
    output_dir: Path, samples: list[FeatureSample], features: tuple[str, ...],
    evidence: list[FeatureEvidence],
) -> None:
    """Write the package except its manifest; for example, the orchestrator adds manifest last."""
    redundancy = calculate_feature_redundancy(samples, features)
    targets = (*features, *(group.name for group in REMOVAL_GROUPS))
    recommendations = build_removal_recommendations(evidence, targets)
    write_feature_redundancy(output_dir / "feature_redundancy.csv", redundancy)
    write_feature_evidence(output_dir / "feature_predictive_evidence.csv", evidence)
    report_path = output_dir / "feature_selection_report.md"
    report_path.write_text(selection_report_markdown(recommendations, redundancy, evidence))
    provisional = provisional_feature_contract(features, recommendations, sha256_file(report_path))
    write_json_artifact(output_dir / "shared_feature_contract.json", provisional)
    save_feature_selection_figures(output_dir, features, redundancy, evidence, recommendations)


def write_json_artifact(path: Path, value: dict[str, object]) -> None:
    """Write deterministic JSON; for example, manifest and contract keys are sorted."""
    serialized = json.dumps(value, indent=2, sort_keys=True)
    terminated = f"{serialized}\n"
    path.write_text(terminated)
