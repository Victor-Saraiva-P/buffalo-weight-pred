"""Freshness and integrity for the provisional approach-selection package."""

from __future__ import annotations

import json
import re
from pathlib import Path

from buffalo_weight.baseline_comparison_artifacts import METRIC_COLUMNS
from buffalo_weight.baseline_comparison_inputs import SOURCES
from buffalo_weight.baseline_comparison_provenance import BaselineComparisonProvenance
from buffalo_weight.feature_selection_manifest import artifact_output_records
from buffalo_weight.hashing import sha256_file
from buffalo_weight.reproduction_config import ReportContract


OUTPUT_FILES = (
    "baseline_metrics.csv", "selected_approach.json", "approach_selection_report.md",
    "global_mae.png", "predicted_vs_observed.png", "residuals_vs_observed.png",
)
VALIDATIONS = [
    "current_baseline_manifests", "one_oof_prediction_per_valid_mask",
    "matching_canonical_folds", "confirmed_feature_contract", "pooled_oof_metrics",
    "signed_residuals", "schemas", "ordering", "sha256", "human_decision_unset",
]
MANIFEST_KEYS = {
    "manifest_version", "package_type", "revision", "stage", "status", "command",
    "recipe_sha256", "dependencies", "inputs", "source_commit", "report_sha256",
    "decision_url", "outputs", "validations",
}
HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def approach_selection_output_dir(contract: ReportContract) -> Path:
    """Locate provisional evidence; for example, confirmed evidence remains elsewhere."""
    stage_name = "approach_selection"
    output_dir = contract.artifacts_root / stage_name
    return output_dir


def comparison_identity(
    contract: ReportContract, provenance: BaselineComparisonProvenance,
) -> dict[str, object]:
    """Build reusable identity; for example, any upstream manifest change invalidates it."""
    return {
        "manifest_version": 1, "package_type": "provisional_evidence", "revision": 1,
        "stage": "approach_selection", "status": "provisional",
        "command": "python main.py compare-baselines",
        "recipe_sha256": provenance.comparison_recipe_hash(),
        "dependencies": provenance.comparison_dependencies(),
        "inputs": baseline_comparison_input_records(contract), "validations": VALIDATIONS,
    }


def baseline_comparison_status(
    contract: ReportContract, provenance: BaselineComparisonProvenance,
) -> str:
    """Classify the package; for example, a modified figure makes it obsolete."""
    output_dir = approach_selection_output_dir(contract)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return "absent"
    try:
        manifest = json.loads(manifest_path.read_text())
        validate_baseline_comparison_manifest(manifest, output_dir, contract, provenance)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "obsolete"
    return "reusable"


def complete_comparison_manifest(
    output_dir: Path, identity: dict[str, object], provenance: BaselineComparisonProvenance,
) -> dict[str, object]:
    """Complete the provisional manifest; for example, output hashes are added last."""
    manifest = identity.copy()
    manifest.update({
        "source_commit": provenance.repository_commit(),
        "report_sha256": sha256_file(output_dir / "approach_selection_report.md"),
        "decision_url": None, "outputs": artifact_output_records(output_dir, OUTPUT_FILES),
    })
    return manifest


def validate_baseline_comparison_manifest(
    manifest: object, output_dir: Path, contract: ReportContract,
    provenance: BaselineComparisonProvenance,
) -> None:
    """Verify current identity and outputs; for example, provisional status is mandatory."""
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        keys = sorted(manifest) if isinstance(manifest, dict) else manifest
        raise ValueError(
            f"comparison manifest keys were {keys!r}; expected {sorted(MANIFEST_KEYS)!r}"
        )
    identity = comparison_identity(contract, provenance)
    actual = {key: manifest.get(key) for key in identity}
    if actual != identity:
        raise ValueError(f"comparison identity was {actual!r}; expected {identity!r}")
    _validate_completion_fields(manifest, output_dir)


def _validate_completion_fields(manifest: dict[str, object], output_dir: Path) -> None:
    commit = manifest.get("source_commit")
    outputs = artifact_output_records(output_dir, OUTPUT_FILES)
    report_hash = sha256_file(output_dir / "approach_selection_report.md")
    actual = (manifest.get("outputs"), manifest.get("report_sha256"),
              manifest.get("decision_url"))
    expected = (outputs, report_hash, None)
    if not isinstance(commit, str) or HEX_COMMIT.fullmatch(commit) is None:
        raise ValueError(f"comparison source commit was {commit!r}; expected a full Git SHA")
    if actual != expected:
        raise ValueError(f"comparison completion was {actual!r}; expected {expected!r}")
    _validate_metric_schema(outputs)


def _validate_metric_schema(outputs: dict[str, dict[str, object]]) -> None:
    record = outputs["baseline_metrics.csv"]
    expected_rows = len(SOURCES) * 8
    actual = (record.get("schema"), record.get("row_count"))
    expected = (METRIC_COLUMNS, expected_rows)
    if actual != expected:
        raise ValueError(f"comparison metrics were {actual!r}; expected {expected!r}")


def baseline_comparison_input_records(contract: ReportContract) -> dict[str, dict[str, object]]:
    """Return the input records for baseline comparison; for example, ``baseline_comparison_input_records(contract)``.

    Example: ``inputs = baseline_comparison_input_records(contract)`` fingerprints all baselines.
    """
    comparison_input_paths: dict[str, Path] = {}
    for source in SOURCES:
        root = contract.artifacts_root / "baselines" / source.directory_name
        comparison_input_paths[f"baselines/{source.directory_name}/manifest.json"] = (
            root / "manifest.json"
        )
        comparison_input_paths[f"baselines/{source.directory_name}/predictions.csv"] = (
            root / "predictions.csv"
        )
    confirmed = contract.confirmed_feature_selection_dir
    comparison_input_paths["confirmed_features/manifest.json"] = confirmed / "manifest.json"
    comparison_input_paths["confirmed_features/shared_feature_contract.json"] = (
        confirmed / "shared_feature_contract.json"
    )
    return {
        name: {"sha256": sha256_file(path)}
        for name, path in comparison_input_paths.items()
    }
