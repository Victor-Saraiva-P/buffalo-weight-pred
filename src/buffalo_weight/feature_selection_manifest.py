"""Freshness and integrity for the provisional feature-selection stage."""

from __future__ import annotations

import json
from pathlib import Path

from buffalo_weight.csv_io import csv_columns, csv_row_count
from buffalo_weight.feature_selection_contract import EVIDENCE_COLUMNS, REDUNDANCY_COLUMNS
from buffalo_weight.hashing import sha256_file
from buffalo_weight.png_artifact import read_png_artifact_spec
from buffalo_weight.feature_selection_provenance import FeatureSelectionProvenance
from buffalo_weight.reproduction_config import ReportContract

OUTPUT_FILES = (
    "feature_redundancy.csv", "feature_predictive_evidence.csv",
    "shared_feature_contract.json", "feature_selection_report.md",
    "redundancy_heatmap.png", "removal_heatmap.png", "permutation_effects.png",
)


def feature_selection_identity(
    contract: ReportContract, provenance: FeatureSelectionProvenance
) -> dict[str, object]:
    """Fingerprint inputs and recipe; for example, input snapshot changes invalidate reuse."""
    input_manifest = contract.inputs_output_dir / "manifest.json"
    return {
        "manifest_version": 1,
        "stage": "feature_selection",
        "inputs": _input_records(input_manifest),
        "recipe_sha256": provenance.feature_selection_recipe_hash(),
        "dependencies": provenance.feature_selection_dependencies(),
    }


def feature_selection_status(
    contract: ReportContract, provenance: FeatureSelectionProvenance
) -> str:
    """Classify the stage; for example, a changed output is obsolete."""
    manifest_path = feature_selection_output_dir(contract) / "manifest.json"
    if not manifest_path.is_file():
        return "absent"
    try:
        manifest = json.loads(manifest_path.read_text())
        return "reusable" if _manifest_current(manifest, contract, provenance) else "obsolete"
    except (OSError, ValueError, TypeError):
        return "obsolete"


def complete_feature_selection_manifest(
    contract: ReportContract, output_dir: Path, identity: dict[str, object],
    source_commit: str,
) -> dict[str, object]:
    """Describe a provisional package; for example, output hashes are captured last."""
    manifest = identity.copy()
    manifest.update({
        "package_type": "reconstructible_stage", "revision": 1,
        "status": "provisional", "source_commit": source_commit,
        "command": "python main.py feature-selection",
        "decision_url": None,
        "report_sha256": sha256_file(output_dir / "feature_selection_report.md"),
        "outputs": _output_records(output_dir),
        "validations": ["schemas", "ordering", "sha256", "experiment_coverage",
                        "human_decision_absent", "figures_300_dpi"],
    })
    return manifest


def validate_feature_selection_manifest(
    manifest: dict[str, object], output_dir: Path
) -> None:
    """Validate the completed package; for example, promotion fields must remain absent."""
    outputs = manifest.get("outputs")
    if manifest.get("status") != "provisional" or not isinstance(outputs, dict):
        raise ValueError(
            f"feature manifest status/outputs were {manifest.get('status')!r}/{outputs!r}; "
            "expected provisional status and output mapping"
        )
    _validate_audit_fields(manifest, output_dir)
    current = _output_records(output_dir)
    if outputs != current:
        raise ValueError(f"feature manifest outputs were {outputs!r}; expected {current!r}")


def _validate_audit_fields(manifest: dict[str, object], output_dir: Path) -> None:
    report_hash = sha256_file(output_dir / "feature_selection_report.md")
    inputs = manifest.get("inputs")
    expected_input_names = {"manifest.json", "feature_index.csv", "canonical_split.csv"}
    actual_input_names = set(inputs) if isinstance(inputs, dict) else set()
    if (manifest.get("decision_url") is not None or manifest.get("report_sha256") != report_hash
            or actual_input_names != expected_input_names):
        raise ValueError(
            f"manifest audit fields were decision={manifest.get('decision_url')!r}, "
            f"report={manifest.get('report_sha256')!r}, inputs={actual_input_names!r}; "
            f"expected null decision, report {report_hash!r}, inputs {expected_input_names!r}"
        )


def _manifest_current(
    manifest: object, contract: ReportContract, provenance: FeatureSelectionProvenance
) -> bool:
    if not isinstance(manifest, dict) or manifest.get("status") != "provisional":
        return False
    identity = feature_selection_identity(contract, provenance)
    if any(manifest.get(key) != value for key, value in identity.items()):
        return False
    outputs = manifest.get("outputs")
    return isinstance(outputs, dict) and outputs == _output_records_if_present(
        feature_selection_output_dir(contract)
    )


def feature_selection_output_dir(contract: ReportContract) -> Path:
    """Locate provisional artifacts; for example, outputs stay beneath generated/report."""
    stage_name = "feature_selection"
    output_dir = contract.artifacts_root / stage_name
    return output_dir


def _output_records_if_present(output_dir: Path) -> dict[str, dict[str, object]]:
    if any(not (output_dir / name).is_file() for name in OUTPUT_FILES):
        return {}
    return _output_records(output_dir)


def _output_records(output_dir: Path) -> dict[str, dict[str, object]]:
    records = {name: _output_record(output_dir / name) for name in OUTPUT_FILES}
    complete_records = records
    return complete_records


def _output_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {"sha256": sha256_file(path)}
    if path.suffix == ".csv":
        record.update({"row_count": csv_row_count(path), "schema": csv_columns(path)})
    elif path.suffix == ".png":
        record.update(_png_schema(path))
    elif path.suffix == ".json":
        loaded = json.loads(path.read_text())
        record.update({"row_count": 1, "schema": sorted(loaded) if isinstance(loaded, dict) else []})
    else:
        record.update({"row_count": len(path.read_text().splitlines()), "schema": "markdown"})
    return record


def _png_schema(path: Path) -> dict[str, object]:
    specification = read_png_artifact_spec(path)
    schema = {"format": "png", "width_px": specification.width_px,
              "height_px": specification.height_px, "dpi": specification.dpi}
    return {"row_count": 1, "schema": schema}


def _input_records(manifest_path: Path) -> dict[str, dict[str, object]]:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("outputs"), dict):
        raise ValueError(f"inputs manifest was {manifest!r}; expected a mapping with output records")
    records = {
        "manifest.json": {"sha256": sha256_file(manifest_path), "row_count": 1,
                          "schema": sorted(manifest)},
    }
    for name in ("feature_index.csv", "canonical_split.csv"):
        records[name] = _normalized_input_record(manifest["outputs"].get(name), name)
    return records


def _normalized_input_record(record: object, name: str) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError(f"inputs output record was {record!r} for {name}; expected a mapping")
    normalized = {"sha256": record.get("sha256"), "row_count": record.get("rows"),
                  "schema": record.get("columns")}
    if None in normalized.values():
        raise ValueError(f"inputs output record was {record!r} for {name}; expected hash/schema/count")
    return normalized


def expected_csv_schemas() -> dict[str, list[str]]:
    """Expose canonical schemas; for example, validation compares exact column order."""
    return {"feature_redundancy.csv": REDUNDANCY_COLUMNS,
            "feature_predictive_evidence.csv": EVIDENCE_COLUMNS}
