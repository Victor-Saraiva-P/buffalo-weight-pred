"""Freshness and integrity for the provisional feature-selection stage."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from buffalo_weight.csv_io import csv_columns, csv_row_count
from buffalo_weight.feature_selection_contract import EVIDENCE_COLUMNS, REDUNDANCY_COLUMNS
from buffalo_weight.hashing import sha256_file
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
        "input_manifest_sha256": sha256_file(input_manifest),
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
    current = _output_records(output_dir)
    if outputs != current:
        raise ValueError(f"feature manifest outputs were {outputs!r}; expected {current!r}")


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
    return contract.artifacts_root / "feature_selection"


def _output_records_if_present(output_dir: Path) -> dict[str, dict[str, object]]:
    if any(not (output_dir / name).is_file() for name in OUTPUT_FILES):
        return {}
    return _output_records(output_dir)


def _output_records(output_dir: Path) -> dict[str, dict[str, object]]:
    return {name: _output_record(output_dir / name) for name in OUTPUT_FILES}


def _output_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {"sha256": sha256_file(path)}
    if path.suffix == ".csv":
        record.update({"rows": csv_row_count(path), "columns": csv_columns(path)})
    elif path.suffix == ".png":
        record.update(_png_schema(path))
    elif path.suffix == ".json":
        loaded = json.loads(path.read_text())
        record["keys"] = sorted(loaded) if isinstance(loaded, dict) else []
    else:
        record["format"] = "markdown"
    return record


def _png_schema(path: Path) -> dict[str, object]:
    with Image.open(path) as figure:
        dpi = figure.info.get("dpi", (0.0, 0.0))
        return {"format": "png", "width_px": figure.width, "height_px": figure.height,
                "dpi": round(float(dpi[0]))}


def expected_csv_schemas() -> dict[str, list[str]]:
    """Expose canonical schemas; for example, validation compares exact column order."""
    return {"feature_redundancy.csv": REDUNDANCY_COLUMNS,
            "feature_predictive_evidence.csv": EVIDENCE_COLUMNS}
