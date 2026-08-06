"""Normalize and validate the five inputs to controlled baseline comparison."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from buffalo_weight.baseline_artifacts import PREDICTION_COLUMNS as RF_COLUMNS
from buffalo_weight.baseline_comparison_types import ComparisonPrediction, EvaluationRole
from buffalo_weight.compact_cnn_artifacts import PREDICTION_COLUMNS as COMPACT_COLUMNS
from buffalo_weight.dense_baseline_artifacts import PREDICTION_COLUMNS as DENSE_COLUMNS
from buffalo_weight.feature_confirmation_manifest import validate_confirmed_feature_package
from buffalo_weight.hashing import sha256_file
from buffalo_weight.input_schema import SPLIT_COLUMNS
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.resnet_baseline_artifacts import PREDICTION_COLUMNS as RESNET_COLUMNS


EXPECTED_COMPARISON_SAMPLE_COUNT = 132


@dataclass(frozen=True)
class _PredictionSource:
    directory_name: str
    configuration: str
    approach: str
    evaluation_role: EvaluationRole
    columns: list[str]
    observed_column: str
    predicted_column: str


SOURCES = (
    _PredictionSource("random_forest_baseline", "random_forest_baseline", "random_forest",
                      "candidate", RF_COLUMNS, "observed_weight_kg", "predicted_weight_kg"),
    _PredictionSource("dense", "dense", "dense_feature_network", "candidate", DENSE_COLUMNS,
                      "observed_weight_kg", "predicted_weight_kg"),
    _PredictionSource("compact_cnn", "compact_cnn", "compact_cnn", "candidate",
                      COMPACT_COLUMNS, "observed_weight_kg", "predicted_weight_kg"),
    _PredictionSource("resnet18_pretrained_partial", "resnet18_pretrained_partial", "resnet18",
                      "candidate", RESNET_COLUMNS, "weight_kg", "prediction_kg"),
    _PredictionSource("training_mean_reference", "training_mean_reference", "training_mean",
                      "reference", RF_COLUMNS, "observed_weight_kg", "predicted_weight_kg"),
)


def load_comparison_predictions(contract: ReportContract) -> list[ComparisonPrediction]:
    """Load current OOF evidence; for example, any incomplete configuration is rejected."""
    _require_canonical_sample_count(contract)
    selected_features = validate_confirmed_feature_package(contract)
    canonical_rows = _canonical_identity(contract.inputs_output_dir / "canonical_split.csv")
    normalized: list[ComparisonPrediction] = []
    for source in SOURCES:
        normalized.extend(_load_source(contract, source, canonical_rows, selected_features))
    return normalized


def _require_canonical_sample_count(contract: ReportContract) -> None:
    count = contract.inputs.expected_mask_count
    if count != EXPECTED_COMPARISON_SAMPLE_COUNT:
        raise ValueError(
            f"comparison sample count was {count!r}; expected exactly "
            f"{EXPECTED_COMPARISON_SAMPLE_COUNT} OOF predictions per configuration"
        )


def _load_source(
    contract: ReportContract, source: _PredictionSource,
    canonical_rows: dict[str, tuple[str, int, float]], selected_features: tuple[str, ...],
) -> list[ComparisonPrediction]:
    output_dir = contract.artifacts_root / "baselines" / source.directory_name
    manifest = _read_manifest(output_dir / "manifest.json", source.configuration)
    prediction_path = output_dir / "predictions.csv"
    rows = _read_exact_csv(prediction_path, source.columns)
    _validate_prediction_record(manifest, prediction_path, source, len(rows))
    _validate_feature_compatibility(manifest, source, selected_features, contract)
    predictions = [_normalized_prediction(row, source) for row in rows]
    _validate_source_predictions(predictions, canonical_rows, source.configuration)
    return predictions


def _read_manifest(path: Path, configuration: str) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{configuration} manifest was unavailable at {path}; expected complete JSON"
        ) from error
    if not isinstance(loaded, dict) or loaded.get("status") != "complete":
        raise ValueError(f"{configuration} manifest was {loaded!r}; expected complete mapping")
    return loaded


def _read_exact_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            columns, rows = list(reader.fieldnames or []), list(reader)
    except OSError as error:
        raise ValueError(f"prediction path was {path}; expected an existing CSV") from error
    if columns != expected_columns:
        raise ValueError(
            f"prediction columns were {columns!r} at {path}; expected {expected_columns!r}"
        )
    return rows


def _validate_prediction_record(
    manifest: dict[str, object], path: Path, source: _PredictionSource, row_count: int,
) -> None:
    outputs = manifest.get("outputs")
    record = outputs.get("predictions.csv") if isinstance(outputs, dict) else None
    actual = _normalized_output_record(record)
    expected = (sha256_file(path), EXPECTED_COMPARISON_SAMPLE_COUNT, source.columns)
    if actual != expected or row_count != EXPECTED_COMPARISON_SAMPLE_COUNT:
        raise ValueError(
            f"{source.configuration} prediction integrity was {actual!r}/{row_count}; "
            f"expected {expected!r}/{EXPECTED_COMPARISON_SAMPLE_COUNT} live rows"
        )


def _normalized_output_record(record: object) -> tuple[object, object, object] | object:
    if not isinstance(record, dict):
        return record
    row_count = record.get("rows", record.get("row_count"))
    schema = record.get("columns", record.get("schema"))
    return record.get("sha256"), row_count, schema


def _validate_feature_compatibility(
    manifest: dict[str, object], source: _PredictionSource, selected_features: tuple[str, ...],
    contract: ReportContract,
) -> None:
    recorded = manifest.get("selected_features")
    if source.configuration == "training_mean_reference":
        if recorded != []:
            raise ValueError(
                f"training_mean_reference selected features were {recorded!r}; expected []"
            )
        return
    if source.configuration in {"random_forest_baseline", "dense"}:
        if recorded != list(selected_features):
            raise ValueError(
                f"{source.configuration} selected features were {recorded!r}; "
                f"expected confirmed contract {list(selected_features)!r}"
            )
        return
    _validate_mask_model_feature_gate(manifest, source, contract)


def _validate_mask_model_feature_gate(
    manifest: dict[str, object], source: _PredictionSource, contract: ReportContract,
) -> None:
    # ResNet-18 is a convolutional mask model reading raw binary masks directly.
    # It operates outside the tabular feature selection report contract.
    if source.configuration == "resnet18_pretrained_partial":
        return
    report_path = contract.confirmed_feature_selection_dir / "feature_selection_report.md"
    recorded = manifest.get("reviewed_report_sha256")
    expected = sha256_file(report_path)
    if recorded != expected:
        raise ValueError(
            f"{source.configuration} feature report was {recorded!r}; expected {expected!r}"
        )


def _canonical_identity(path: Path) -> dict[str, tuple[str, int, float]]:
    rows = _read_exact_csv(path, SPLIT_COLUMNS)
    identity = {
        row["file_name"]: (
            row["weight_category"], _integer(row.get("fold"), "fold"),
            _finite_number(row.get("weight_kg"), "weight_kg"),
        ) for row in rows
    }
    if len(identity) != EXPECTED_COMPARISON_SAMPLE_COUNT:
        raise ValueError(
            f"canonical split identities were {len(identity)!r}; "
            f"expected {EXPECTED_COMPARISON_SAMPLE_COUNT} unique rows"
        )
    return identity


def _normalized_prediction(
    row: dict[str, str], source: _PredictionSource,
) -> ComparisonPrediction:
    observed = _finite_number(row.get(source.observed_column), source.observed_column)
    predicted = _finite_number(row.get(source.predicted_column), source.predicted_column)
    return ComparisonPrediction(
        source.configuration, source.approach, source.evaluation_role, row["file_name"],
        row["weight_category"], _integer(row.get("fold"), "fold"), observed, predicted,
    )


def _validate_source_predictions(
    predictions: list[ComparisonPrediction], canonical: dict[str, tuple[str, int, float]],
    configuration: str,
) -> None:
    names = [row.file_name for row in predictions]
    if names != sorted(canonical):
        raise ValueError(
            f"{configuration} prediction names were {names!r}; expected {sorted(canonical)!r}"
        )
    mismatches = [
        row.file_name for row in predictions
        if (row.weight_category, row.fold, row.observed_weight_kg) != canonical[row.file_name]
    ]
    if mismatches:
        raise ValueError(
            f"{configuration} fold/label identities differed for {mismatches!r}; "
            "expected the canonical split"
        )


def _finite_number(candidate: object, field: str) -> float:
    try:
        parsed = float(str(candidate))
    except ValueError as error:
        raise ValueError(f"{field} was {candidate!r}; expected finite numeric text") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} was {candidate!r}; expected finite numeric text")
    return parsed


def _integer(candidate: object, field: str) -> int:
    try:
        return int(str(candidate))
    except ValueError as error:
        raise ValueError(f"{field} was {candidate!r}; expected integer text") from error
