"""Public CSV schemas and serialization for compact-CNN OOF evidence."""

from __future__ import annotations

import csv
from pathlib import Path

from buffalo_weight.compact_cnn_evaluation import CompactCnnEvaluation
from buffalo_weight.csv_io import format_csv_number
from buffalo_weight.hashing import sha256_file


PREDICTION_COLUMNS = [
    "model", "file_name", "farm", "weight_category", "fold", "observed_weight_kg",
    "predicted_weight_kg", "residual_kg", "absolute_error_kg",
]
METRIC_COLUMNS = [
    "model", "scope", "group", "fold", "sample_count", "mae_kg", "rmse_kg",
    "bias_kg", "r2",
]
OUTPUT_NAMES = ("predictions.csv", "fold_metrics.csv")


def write_compact_cnn_artifacts(
    output_dir: Path, evaluation: CompactCnnEvaluation,
) -> None:
    """Write public tables; for example, the caller writes the manifest last."""
    _write_csv(output_dir / OUTPUT_NAMES[0], PREDICTION_COLUMNS,
               _prediction_rows(evaluation))
    _write_csv(output_dir / OUTPUT_NAMES[1], METRIC_COLUMNS, _metric_rows(evaluation))


def compact_cnn_output_records(output_dir: Path) -> dict[str, dict[str, object]]:
    """Describe table outputs; for example, manifests record schema, count, and hash."""
    records = {}
    for name in OUTPUT_NAMES:
        path = output_dir / name
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.reader(source))
        records[name] = {"sha256": sha256_file(path), "schema": rows[0],
                         "row_count": len(rows) - 1}
    return records


def validate_compact_cnn_output_tables(output_dir: Path, expected_count: int) -> None:
    """Validate completion tables; for example, predictions cover every valid mask."""
    records = compact_cnn_output_records(output_dir)
    prediction, metrics = records[OUTPUT_NAMES[0]], records[OUTPUT_NAMES[1]]
    if prediction["schema"] != PREDICTION_COLUMNS or prediction["row_count"] != expected_count:
        raise ValueError(
            f"compact CNN predictions were {prediction!r}; "
            f"expected schema and {expected_count} rows"
        )
    _validate_metric_record(metrics)


def _validate_metric_record(metrics: dict[str, object]) -> None:
    metric_count = metrics["row_count"]
    valid = (metrics["schema"] == METRIC_COLUMNS and isinstance(metric_count, int)
             and metric_count >= 2)
    if not valid:
        raise ValueError(
            f"compact CNN metrics were {metrics!r}; expected canonical schema and rows"
        )


def _prediction_rows(evaluation: CompactCnnEvaluation) -> list[dict[str, str]]:
    rows = []
    for prediction in evaluation.predictions:
        residual = prediction.predicted_weight_kg - prediction.observed_weight_kg
        rows.append({
            "model": "compact_cnn", "file_name": prediction.file_name,
            "farm": prediction.farm, "weight_category": prediction.weight_category,
            "fold": str(prediction.fold),
            "observed_weight_kg": format_csv_number(prediction.observed_weight_kg),
            "predicted_weight_kg": format_csv_number(prediction.predicted_weight_kg),
            "residual_kg": format_csv_number(residual),
            "absolute_error_kg": format_csv_number(abs(residual)),
        })
    return rows


def _metric_rows(evaluation: CompactCnnEvaluation) -> list[dict[str, str]]:
    return [{
        "model": "compact_cnn", "scope": metric.scope, "group": metric.group,
        "fold": "" if metric.fold is None else str(metric.fold),
        "sample_count": str(metric.sample_count), "mae_kg": format_csv_number(metric.mae_kg),
        "rmse_kg": format_csv_number(metric.rmse_kg),
        "bias_kg": format_csv_number(metric.bias_kg), "r2": format_csv_number(metric.r2),
    } for metric in evaluation.metrics]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
