"""Schemas, metrics and deterministic I/O for one ResNet-18 evaluation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from buffalo_weight.csv_io import format_csv_number
from buffalo_weight.hashing import sha256_file
from buffalo_weight.input_schema import SPLIT_COLUMNS
from buffalo_weight.resnet_baseline_evaluation import ResNetOofPrediction, ResNetSample


MODEL_CONFIG = "resnet18_pretrained_partial"
PREDICTION_COLUMNS = [
    "model_config", "fold", "file_name", "weight_category", "weight_kg",
    "prediction_kg", "residual_kg", "absolute_error_kg",
]
METRIC_COLUMNS = [
    "model_config", "scope", "fold", "n", "mae_kg", "rmse_kg", "bias_kg", "r2",
]


def load_resnet_samples(split_path: Path, masks_dir: Path) -> tuple[ResNetSample, ...]:
    """Load canonical rows; for example, each sample resolves its curated PNG."""
    rows = _read_csv(split_path, SPLIT_COLUMNS)
    samples = tuple(_sample(row, masks_dir) for row in rows)
    names = [sample.file_name for sample in samples]
    if len(names) != len(set(names)):
        raise ValueError(f"baseline file names were {names!r}; expected unique names")
    return tuple(sorted(samples, key=lambda sample: sample.file_name))


def validate_predictions(
    predictions: list[ResNetOofPrediction], samples: tuple[ResNetSample, ...]
) -> None:
    """Validate OOF coverage; for example, duplicates or missing masks fail atomically."""
    expected = {sample.file_name: sample for sample in samples}
    names = [prediction.file_name for prediction in predictions]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError(
            f"prediction names were {names!r}; expected one row for each {sorted(expected)!r}"
        )
    for prediction in predictions:
        _validate_prediction(prediction, expected[prediction.file_name])


def write_resnet_outputs(
    output_dir: Path, predictions: list[ResNetOofPrediction]
) -> None:
    """Write tidy outputs; for example, OOF metrics follow the five fold rows."""
    ordered = sorted(predictions, key=lambda row: row.file_name)
    _write_csv(output_dir / "predictions.csv", PREDICTION_COLUMNS,
               [_prediction_row(row) for row in ordered])
    _write_csv(output_dir / "metrics.csv", METRIC_COLUMNS, _metric_rows(ordered))


def output_metadata(output_dir: Path) -> dict[str, dict[str, object]]:
    """Describe complete outputs; for example, manifests bind hashes and schemas."""
    return {
        "metrics.csv": _artifact_record(output_dir / "metrics.csv", METRIC_COLUMNS),
        "predictions.csv": _artifact_record(
            output_dir / "predictions.csv", PREDICTION_COLUMNS
        ),
    }


def validate_output_metadata(
    output_dir: Path, outputs: object, expected_predictions: int, expected_folds: int,
) -> None:
    """Validate manifest-bound outputs; for example, tampering makes reuse obsolete."""
    if not isinstance(outputs, dict):
        raise ValueError(f"baseline outputs were {outputs!r}; expected a mapping")
    expected = output_metadata(output_dir)
    if outputs != expected:
        raise ValueError(f"baseline outputs were {outputs!r}; expected {expected!r}")
    prediction_rows = _read_csv(output_dir / "predictions.csv", PREDICTION_COLUMNS)
    metric_rows = _read_csv(output_dir / "metrics.csv", METRIC_COLUMNS)
    if len(prediction_rows) != expected_predictions or len(metric_rows) != expected_folds + 1:
        raise ValueError(
            f"baseline row counts were {len(prediction_rows)}/{len(metric_rows)}; "
            f"expected {expected_predictions}/{expected_folds + 1}"
        )


def _sample(row: dict[str, str], masks_dir: Path) -> ResNetSample:
    name = _required_text(row, "file_name", "canonical split row")
    mask_path = masks_dir / name
    if not mask_path.is_file():
        raise ValueError(f"mask path was {mask_path}; expected an existing curated mask")
    category = _required_text(row, "weight_category", name)
    return ResNetSample(name, mask_path, category, _integer(row, "fold", name),
                        _finite_number(row, "weight_kg", name))


def _validate_prediction(prediction: ResNetOofPrediction, sample: ResNetSample) -> None:
    observed = (prediction.fold, prediction.weight_category, prediction.weight_kg)
    expected = (sample.fold, sample.weight_category, sample.weight_kg)
    if observed != expected or not math.isfinite(prediction.prediction_kg):
        raise ValueError(
            f"prediction identity/value was {observed!r}/{prediction.prediction_kg!r} "
            f"for {sample.file_name!r}; expected {expected!r} and a finite prediction"
        )


def _prediction_row(prediction: ResNetOofPrediction) -> dict[str, str]:
    residual = prediction.prediction_kg - prediction.weight_kg
    return {
        "model_config": MODEL_CONFIG, "fold": str(prediction.fold),
        "file_name": prediction.file_name, "weight_category": prediction.weight_category,
        "weight_kg": format_csv_number(prediction.weight_kg),
        "prediction_kg": format_csv_number(prediction.prediction_kg),
        "residual_kg": format_csv_number(residual),
        "absolute_error_kg": format_csv_number(abs(residual)),
    }


def _metric_rows(predictions: list[ResNetOofPrediction]) -> list[dict[str, str]]:
    rows = []
    for fold in sorted({prediction.fold for prediction in predictions}):
        fold_predictions = [row for row in predictions if row.fold == fold]
        rows.append(_metric_row("fold", str(fold), fold_predictions))
    rows.append(_metric_row("oof", "", predictions))
    return rows


def _metric_row(
    scope: str, fold: str, predictions: list[ResNetOofPrediction]
) -> dict[str, str]:
    observed = np.asarray([row.weight_kg for row in predictions], dtype=np.float64)
    estimated = np.asarray([row.prediction_kg for row in predictions], dtype=np.float64)
    residuals = estimated - observed
    return {
        "model_config": MODEL_CONFIG, "scope": scope, "fold": fold,
        "n": str(len(predictions)), "mae_kg": format_csv_number(np.mean(np.abs(residuals))),
        "rmse_kg": format_csv_number(np.sqrt(np.mean(np.square(residuals)))),
        "bias_kg": format_csv_number(np.mean(residuals)),
        "r2": format_csv_number(_r2_score(observed, estimated)),
    }


def _r2_score(observed: np.ndarray, estimated: np.ndarray) -> float:
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    if denominator == 0.0:
        return 0.0
    numerator = float(np.sum(np.square(observed - estimated)))
    return 1.0 - numerator / denominator


def _artifact_record(path: Path, columns: list[str]) -> dict[str, object]:
    rows = _read_csv(path, columns)
    return {"sha256": sha256_file(path), "row_count": len(rows), "schema": columns}


def _read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if fields != columns:
        raise ValueError(f"CSV columns were {fields!r} at {path}; expected exactly {columns!r}")
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Write deterministic JSON last; for example, success becomes visible atomically."""
    path.write_text(f"{json.dumps(manifest, indent=2, sort_keys=True)}\n")


def _required_text(row: dict[str, str], field: str, context: str) -> str:
    value = row.get(field)
    if not value:
        raise ValueError(f"{field} was {value!r} for {context!r}; expected non-empty text")
    return value


def _finite_number(row: dict[str, str], field: str, context: str) -> float:
    value = row.get(field)
    try:
        parsed = float(str(value))
    except ValueError as error:
        raise ValueError(f"{field} was {value!r} for {context!r}; expected finite numeric text") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} was {value!r} for {context!r}; expected finite numeric text")
    return parsed


def _integer(row: dict[str, str], field: str, context: str) -> int:
    value = row.get(field)
    try:
        return int(str(value))
    except ValueError as error:
        raise ValueError(f"{field} was {value!r} for {context!r}; expected integer text") from error
