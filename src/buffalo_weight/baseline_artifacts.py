"""Deterministic serialization for per-configuration baseline artifacts."""

from __future__ import annotations

from pathlib import Path

from buffalo_weight.baseline_types import BaselinePrediction
from buffalo_weight.baseline_metrics import MetricSummary, fold_summaries, grouped_summaries
from buffalo_weight.csv_io import format_csv_number, write_csv_rows


PREDICTION_COLUMNS = [
    "configuration", "evaluation_role", "file_name", "weight_category", "fold",
    "observed_weight_kg", "predicted_weight_kg", "residual_kg", "absolute_error_kg",
]
FOLD_METRIC_COLUMNS = [
    "configuration", "evaluation_role", "fold", "n", "mae_kg", "rmse_kg", "bias_kg", "r2",
]
GROUPED_METRIC_COLUMNS = [
    "configuration", "evaluation_role", "population", "n", "mae_kg", "rmse_kg",
    "bias_kg", "r2",
]


def write_baseline_predictions(path: Path, predictions: list[BaselinePrediction]) -> None:
    """Write one OOF row per mask; for example, rows are ordered by ``file_name``."""
    ordered = sorted(predictions, key=lambda row: row.file_name)
    write_csv_rows([_prediction_record(row) for row in ordered], path, PREDICTION_COLUMNS)


def write_baseline_metrics(output_dir: Path, predictions: list[BaselinePrediction]) -> None:
    """Write fold and pooled metrics; for example, pooled metrics are never fold averages."""
    configuration, role = _identity(predictions)
    fold_rows = [
        _metric_record(configuration, role, "fold", str(fold), summary)
        for fold, summary in fold_summaries(predictions)
    ]
    grouped_rows = [
        _metric_record(configuration, role, "population", population, summary)
        for population, summary in grouped_summaries(predictions)
    ]
    write_csv_rows(fold_rows, output_dir / "fold_metrics.csv", FOLD_METRIC_COLUMNS)
    write_csv_rows(grouped_rows, output_dir / "grouped_metrics.csv", GROUPED_METRIC_COLUMNS)


def _prediction_record(row: BaselinePrediction) -> dict[str, str]:
    return {
        "configuration": row.configuration,
        "evaluation_role": row.evaluation_role,
        "file_name": row.file_name,
        "weight_category": row.weight_category,
        "fold": str(row.fold),
        "observed_weight_kg": format_csv_number(row.observed_weight_kg),
        "predicted_weight_kg": format_csv_number(row.predicted_weight_kg),
        "residual_kg": format_csv_number(row.residual_kg),
        "absolute_error_kg": format_csv_number(row.absolute_error_kg),
    }


def _identity(predictions: list[BaselinePrediction]) -> tuple[str, str]:
    identities = {(row.configuration, row.evaluation_role) for row in predictions}
    if len(identities) != 1:
        raise ValueError(f"prediction identities were {identities!r}; expected exactly one")
    return next(iter(identities))


def _metric_record(
    configuration: str, role: str, dimension: str, value: str, summary: MetricSummary,
) -> dict[str, str]:
    record = {
        "configuration": configuration, "evaluation_role": role,
        "n": str(summary.n), "mae_kg": format_csv_number(summary.mae_kg),
        "rmse_kg": format_csv_number(summary.rmse_kg),
        "bias_kg": format_csv_number(summary.bias_kg), "r2": format_csv_number(summary.r2),
    }
    record[dimension] = value
    return record
