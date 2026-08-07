"""Atomic stage execution for descriptive expanded diagnosis characterization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from buffalo_weight.baseline_comparison_inputs import load_comparison_predictions
from buffalo_weight.baseline_comparison_types import ComparisonPrediction
from buffalo_weight.csv_io import write_csv_rows
from buffalo_weight.diagnostic_artifacts import write_descriptive_diagnostic_artifacts
from buffalo_weight.diagnostic_coverage import DiagnosticCoverageSample
from buffalo_weight.diagnostic_descriptive_slice import (
    DescriptiveDiagnosticSlice,
    build_descriptive_diagnostic_slice,
)
from buffalo_weight.diagnostic_stratified import DiagnosticPrediction
from buffalo_weight.input_schema import SPLIT_COLUMNS
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.snapshot_io import (
    FilesystemSnapshotPublisher,
    SnapshotPublisher,
    clean_snapshot_stage,
)


def diagnostic_descriptive_output_dir(contract: ReportContract) -> Path:
    """Locate output directory for descriptive expanded diagnosis.

    Example: ``diagnostic_descriptive_output_dir(contract)`` returns Path under artifacts_root.
    """
    return contract.artifacts_root / "diagnostics" / "descriptive"


def run_diagnostic_descriptive_stage(
    contract: ReportContract,
    dry_run: bool = False,
    publisher: SnapshotPublisher | None = None,
) -> str:
    """Run the descriptive diagnostic stage.

    Example: ``run_diagnostic_descriptive_stage(contract, dry_run=True)`` checks stage.
    """
    output_dir = diagnostic_descriptive_output_dir(contract)
    manifest_path = output_dir / "manifest.json"
    if dry_run:
        return "reusable" if manifest_path.is_file() else "reconstructible"
    resolved_publisher = publisher or FilesystemSnapshotPublisher()
    clean_snapshot_stage(output_dir)
    coverage_samples, predictions = load_diagnostic_inputs(contract)
    slice_data = build_descriptive_diagnostic_slice(coverage_samples, predictions)
    write_descriptive_diagnostic_artifacts(output_dir, slice_data)
    manifest = {
        "status": "complete",
        "sample_count": slice_data.coverage_summary.sample_count,
        "shared_hard_case_count": len(slice_data.shared_hard_cases),
        "divergent_case_count": len(slice_data.divergent_cases),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return "rebuilt"


def load_diagnostic_inputs(
    contract: ReportContract,
) -> tuple[list[DiagnosticCoverageSample], list[DiagnosticPrediction]]:
    """Load coverage samples and combined baseline/tuned predictions for diagnosis.

    Example: ``load_diagnostic_inputs(contract)`` returns (samples, predictions).
    """
    split_rows = _load_canonical_split_rows(contract.inputs_output_dir / "canonical_split.csv")
    resolutions = _resolve_mask_resolutions(contract.inputs.masks_dir, split_rows)
    samples = _build_coverage_samples(split_rows, resolutions)
    comparison_preds = load_comparison_predictions(contract)
    preds = _build_candidate_predictions(comparison_preds, samples, resolutions)
    preds.extend(_load_tuned_predictions_if_available(contract, samples, resolutions))
    return samples, preds


def _load_canonical_split_rows(split_path: Path) -> list[dict[str, str]]:
    if not split_path.is_file():
        raise ValueError(f"split file was unavailable at {split_path}; expected canonical_split.csv")
    return _read_csv_dict(split_path)


def _build_coverage_samples(
    split_rows: list[dict[str, str]], resolutions: dict[str, str]
) -> list[DiagnosticCoverageSample]:
    return [
        DiagnosticCoverageSample(
            file_name=row["file_name"],
            farm=row["farm"],
            weight_category=row["weight_category"],
            resolution=resolutions[row["file_name"]],
            weight_kg=float(row["weight_kg"]),
        )
        for row in split_rows
    ]


def _build_candidate_predictions(
    comparison_preds: list[ComparisonPrediction],
    samples: list[DiagnosticCoverageSample],
    resolutions: dict[str, str],
) -> list[DiagnosticPrediction]:
    farm_by_file = {s.file_name: s.farm for s in samples}
    return [
        DiagnosticPrediction(
            configuration=p.configuration,
            evaluation_role="baseline" if p.evaluation_role == "candidate" else "reference",
            file_name=p.file_name,
            weight_category=p.weight_category,
            farm=farm_by_file[p.file_name],
            resolution=resolutions[p.file_name],
            observed_weight_kg=p.observed_weight_kg,
            predicted_weight_kg=p.predicted_weight_kg,
        )
        for p in comparison_preds
        if p.evaluation_role == "candidate"
    ]


def _read_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _resolve_mask_resolutions(
    masks_dir: Path, split_rows: list[dict[str, str]]
) -> dict[str, str]:
    resolutions: dict[str, str] = {}
    for row in split_rows:
        file_name = row["file_name"]
        mask_path = masks_dir / file_name
        if not mask_path.is_file():
            raise ValueError(f"mask file was unavailable at {mask_path}; expected existing PNG")
        with Image.open(mask_path) as img:
            resolutions[file_name] = f"{img.width}x{img.height}"
    return resolutions


def _load_tuned_predictions_if_available(
    contract: ReportContract,
    samples: list[DiagnosticCoverageSample],
    resolutions: dict[str, str],
) -> list[DiagnosticPrediction]:
    tuning_dir = contract.artifacts_root / "tuning"
    manifest_path = tuning_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        return []
    # If tuned predictions exist in tuning artifact directory or report
    return []
