"""Manifest construction and validation for configuration tuning evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path

from buffalo_weight.feature_selection_manifest import artifact_output_records
from buffalo_weight.hashing import sha256_file
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.tuning_provenance import TuningProvenance
from buffalo_weight.tuning_types import TuningVariation

OUTPUT_FILES = ("tuning_metrics.csv", "tuning_report.md")
VALIDATIONS = (
    "approach_gate", "pre_registered_variations", "frozen_features",
    "canonical_split", "isolated_tuning",
)
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def tuning_output_dir(contract: ReportContract) -> Path:
    """Locate tuning output directory.

    Example: ``tuning_output_dir(contract)`` points under generated/report/tuning.
    """
    return contract.artifacts_root / "tuning"


def tuning_stage_status(
    contract: ReportContract, provenance: TuningProvenance,
) -> str:
    """Check freshness of tuning stage artifacts.

    Example: missing manifest returns ``absent``.
    """
    output_dir = tuning_output_dir(contract)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return "absent"
    try:
        manifest = json.loads(manifest_path.read_text())
        validate_tuning_manifest(manifest, output_dir, contract, provenance)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "obsolete"
    status_str: str = str(manifest.get("status"))
    if status_str == "baseline_maintained":
        return "baseline_maintained"
    return "reusable"


def build_tuning_manifest(
    output_dir: Path, contract: ReportContract, selected_approach: str,
    baseline_config: str, budget: int, variations: tuple[TuningVariation, ...],
    provenance: TuningProvenance,
) -> dict[str, object]:
    """Build root tuning manifest.

    Example: records selected approach, pre-registered recipes, and output hashes.
    """
    status = "baseline_maintained" if not variations else "complete"
    output_records = artifact_output_records(output_dir, OUTPUT_FILES) if variations else {}
    return {
        "manifest_version": 1,
        "package_type": "tuning_evidence",
        "revision": 1,
        "status": status,
        "command": "python main.py tuning",
        "source_commit": provenance.repository_commit(),
        "tuning_recipe_hash": provenance.tuning_recipe_hash(),
        "dependencies": provenance.tuning_dependencies(),
        "selected_approach": {
            "approach": selected_approach,
            "baseline_configuration": baseline_config,
            "maximum_tuning_variations": budget,
        },
        "pre_registered_variations": [
            {"name": v.name, "recipe_type": type(v.recipe).__name__} for v in variations
        ],
        "outputs": output_records,
        "validations": list(VALIDATIONS),
    }


def validate_tuning_manifest(
    manifest: dict[str, object], output_dir: Path, contract: ReportContract,
    provenance: TuningProvenance,
) -> None:
    """Validate root tuning manifest.

    Example: tampered outputs or commits raise a descriptive ValueError.
    """
    _validate_manifest_types(manifest)
    _validate_manifest_fixed(manifest)
    _validate_source_commit(manifest, provenance)
    _validate_recipe_hash(manifest, provenance)
    _validate_manifest_outputs(manifest, output_dir)


def _validate_manifest_types(manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest was {manifest!r}; expected a mapping")


def _validate_manifest_fixed(manifest: dict[str, object]) -> None:
    fixed = (
        manifest.get("manifest_version"), manifest.get("package_type"),
        manifest.get("revision"), manifest.get("command"), manifest.get("validations"),
    )
    expected = (1, "tuning_evidence", 1, "python main.py tuning", list(VALIDATIONS))
    if fixed != expected:
        raise ValueError(f"manifest fixed fields were {fixed!r}; expected {expected!r}")


def _validate_source_commit(manifest: dict[str, object], provenance: TuningProvenance) -> None:
    commit = manifest.get("source_commit")
    expected = provenance.repository_commit()
    if commit != expected or not isinstance(commit, str) or GIT_SHA.fullmatch(commit) is None:
        raise ValueError(f"tuning source commit was {commit!r}; expected {expected!r}")


def _validate_recipe_hash(manifest: dict[str, object], provenance: TuningProvenance) -> None:
    recipe_hash = manifest.get("tuning_recipe_hash")
    expected = provenance.tuning_recipe_hash()
    if recipe_hash != expected:
        raise ValueError(f"tuning recipe hash was {recipe_hash!r}; expected {expected!r}")


def _validate_manifest_outputs(manifest: dict[str, object], output_dir: Path) -> None:
    if manifest.get("status") == "baseline_maintained":
        return
    expected_outputs = artifact_output_records(output_dir, OUTPUT_FILES)
    if manifest.get("outputs") != expected_outputs:
        raise ValueError(
            f"tuning manifest outputs were {manifest.get('outputs')!r}; expected {expected_outputs!r}"
        )
