"""Manifest construction for one baseline configuration."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from buffalo_weight.baseline_artifacts import (
    FOLD_METRIC_COLUMNS,
    GROUPED_METRIC_COLUMNS,
    PREDICTION_COLUMNS,
)
from buffalo_weight.baseline_provenance import BaselineProvenance
from buffalo_weight.baseline_types import (
    BASELINE_VALIDATIONS,
    BaselineConfiguration,
    BaselineStatus,
    EvaluationRole,
    baseline_definition,
)
from buffalo_weight.csv_io import csv_columns, csv_row_count
from buffalo_weight.feature_confirmation_manifest import SOURCE_MANIFEST_NAME
from buffalo_weight.hashing import sha256_file
from buffalo_weight.reproduction_config import ReportContract, contract_identity


OUTPUT_SCHEMAS = {
    "predictions.csv": PREDICTION_COLUMNS,
    "fold_metrics.csv": FOLD_METRIC_COLUMNS,
    "grouped_metrics.csv": GROUPED_METRIC_COLUMNS,
}
MANIFEST_KEYS = {
    "manifest_version", "package_type", "stage", "status", "configuration",
    "evaluation_role", "command", "source_commit", "recipe_sha256", "dependencies",
    "selected_features", "fold_seed", "training_seed", "report_contract", "inputs",
    "outputs", "validations",
}
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def complete_baseline_manifest(
    contract: ReportContract, output_dir: Path, configuration: BaselineConfiguration,
    role: EvaluationRole, features: tuple[str, ...], provenance: BaselineProvenance,
) -> dict[str, object]:
    """Describe one complete artifact; for example, its output hashes are written last."""
    manifest = baseline_identity(contract, configuration, role, features, provenance)
    manifest.update({
        "source_commit": provenance.repository_commit(),
        "outputs": _output_records(output_dir),
    })
    return manifest


def baseline_configuration_status(
    contract: ReportContract, configuration: BaselineConfiguration,
    role: EvaluationRole, features: tuple[str, ...], provenance: BaselineProvenance,
) -> BaselineStatus:
    """Classify one artifact; for example, an RF recipe edit leaves the reference reusable."""
    output_dir = contract.artifacts_root / "baselines" / configuration
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return "absent"
    try:
        loaded = json.loads(manifest_path.read_text())
        expected = baseline_identity(contract, configuration, role, features, provenance)
        if not isinstance(loaded, dict) or not _manifest_shape_is_valid(loaded):
            return "obsolete"
        if not _identity_matches(loaded, expected):
            return "obsolete"
        return "reusable" if _outputs_match(loaded, output_dir, contract) else "obsolete"
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return "obsolete"


def baseline_identity(
    contract: ReportContract, configuration: BaselineConfiguration,
    role: EvaluationRole, features: tuple[str, ...], provenance: BaselineProvenance,
) -> dict[str, object]:
    """Build freshness identity; for example, output hashes are deliberately separate."""
    definition = baseline_definition(configuration)
    consumed_features = features if definition.consumes_confirmed_features else ()
    return {
        "manifest_version": 1, "package_type": "reconstructible_configuration",
        "stage": "baselines", "status": "complete", "configuration": configuration,
        "evaluation_role": role, "command": "python main.py baselines",
        "recipe_sha256": provenance.baseline_recipe_hash(configuration),
        "dependencies": provenance.baseline_dependencies(configuration),
        "selected_features": list(consumed_features), "fold_seed": contract.inputs.fold_seed,
        "training_seed": 44, "report_contract": contract_identity(contract),
        "inputs": _input_records(contract, definition.consumes_confirmed_features, features),
        "validations": BASELINE_VALIDATIONS,
    }


def _input_records(
    contract: ReportContract, consumes_confirmed_features: bool,
    features: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    consumed_features = features if consumes_confirmed_features else ()
    records = {
        "feature_index.csv": _csv_projection_record(
            contract.inputs_output_dir / "feature_index.csv",
            ("file_name", "weight_kg", *consumed_features),
        ),
        "canonical_split.csv": _csv_projection_record(
            contract.inputs_output_dir / "canonical_split.csv",
            ("file_name", "weight_category", "fold"),
        ),
    }
    paths = _confirmed_input_paths(contract) if consumes_confirmed_features else {}
    records.update({name: {"sha256": sha256_file(path)} for name, path in paths.items()})
    return records


def _confirmed_input_paths(contract: ReportContract) -> dict[str, Path]:
    confirmed = contract.confirmed_feature_selection_dir
    return {
        "shared_feature_contract.json": confirmed / "shared_feature_contract.json",
        "confirmed_manifest.json": confirmed / "manifest.json",
        "source_feature_selection_manifest.json": confirmed / SOURCE_MANIFEST_NAME,
    }


def _csv_projection_record(path: Path, columns: tuple[str, ...]) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"CSV columns at {path} omitted {missing!r}; expected consumed columns {columns!r}"
            )
        rows = [{column: row[column] for column in columns} for row in reader]
    ordered = sorted(rows, key=lambda row: row["file_name"])
    serialized = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "rows": len(ordered), "columns": list(columns)}


def _output_records(output_dir: Path) -> dict[str, dict[str, object]]:
    return {
        name: {
            "sha256": sha256_file(output_dir / name),
            "rows": csv_row_count(output_dir / name),
            "columns": csv_columns(output_dir / name),
        }
        for name, columns in OUTPUT_SCHEMAS.items()
    }


def _identity_matches(manifest: dict[str, object], expected: dict[str, object]) -> bool:
    # Extra completion fields are validated separately from freshness identity.
    matches = all(manifest.get(key) == value for key, value in expected.items())
    return matches


def _manifest_shape_is_valid(manifest: dict[str, object]) -> bool:
    source_commit = manifest.get("source_commit")
    commit_valid = isinstance(source_commit, str) and GIT_SHA.fullmatch(source_commit) is not None
    return set(manifest) == MANIFEST_KEYS and commit_valid


def _outputs_match(
    manifest: dict[str, object], output_dir: Path, contract: ReportContract,
) -> bool:
    expected_rows = {
        "predictions.csv": contract.inputs.expected_mask_count,
        "fold_metrics.csv": contract.inputs.fold_count,
        "grouped_metrics.csv": 3,
    }
    if any(not (output_dir / name).is_file() for name in OUTPUT_SCHEMAS):
        return False
    records = _output_records(output_dir)
    schemas_match = all(records[name]["columns"] == columns
                        for name, columns in OUTPUT_SCHEMAS.items())
    counts_match = all(records[name]["rows"] == expected_rows[name] for name in OUTPUT_SCHEMAS)
    return schemas_match and counts_match and manifest.get("outputs") == records
