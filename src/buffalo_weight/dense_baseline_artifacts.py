"""Tidy CSV artifacts and metrics for the dense baseline."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.csv_io import format_csv_number, write_csv_rows
from buffalo_weight.dense_baseline_evaluation import (
    DenseBaselineEvaluation,
    DenseFoldAudit,
    DenseOofPrediction,
)
from buffalo_weight.feature_baselines import DENSE_BASELINE_RECIPE
from buffalo_weight.feature_evaluation import FeatureSample

PREDICTION_COLUMNS = [
    "model_config", "approach", "file_name", "fold", "weight_category",
    "observed_weight_kg", "predicted_weight_kg", "residual_kg", "absolute_error_kg",
]
METRIC_COLUMNS = [
    "model_config", "approach", "scope", "fold", "population", "n",
    "selected_epochs", "mae_kg", "rmse_kg", "bias_kg", "r2",
]
MODEL_CONFIG = "dense"
APPROACH = "dense_feature_network"
Population = Literal["all", "B1", "B10"]
MetricScope = Literal["fold", "oof"]


@dataclass(frozen=True)
class DenseMetric:
    scope: MetricScope
    fold: int | None
    population: Population
    n: int
    selected_epochs: int | None
    mae_kg: float
    rmse_kg: float
    bias_kg: float
    r2: float | None


def write_dense_baseline_artifacts(
    output_dir: Path, evaluation: DenseBaselineEvaluation,
) -> None:
    """Write deterministic public tables; for example, prediction rows sort by filename."""
    # Metrics must be exactly reconstructible from the public six-decimal OOF table.
    predictions = sorted((_public_prediction(row) for row in evaluation.predictions),
                         key=lambda row: row.file_name)
    public_evaluation = DenseBaselineEvaluation(tuple(predictions), evaluation.fold_audits)
    metrics = dense_metrics(public_evaluation)
    write_csv_rows([_prediction_row(row) for row in predictions],
                   output_dir / "predictions.csv", PREDICTION_COLUMNS)
    write_csv_rows([_metric_row(row) for row in metrics],
                   output_dir / "fold_metrics.csv", METRIC_COLUMNS)


def _public_prediction(prediction: DenseOofPrediction) -> DenseOofPrediction:
    rounded = float(format_csv_number(prediction.predicted_weight_kg))
    return DenseOofPrediction(
        prediction.file_name, prediction.fold, prediction.weight_category,
        prediction.observed_weight_kg, rounded,
    )


def dense_metrics(evaluation: DenseBaselineEvaluation) -> list[DenseMetric]:
    """Calculate fold and grouped OOF metrics; for example, OOF is never a fold average."""
    epoch_by_fold = {audit.fold: audit.selected_epochs for audit in evaluation.fold_audits}
    folds = sorted({prediction.fold for prediction in evaluation.predictions})
    fold_rows = [
        _metric("fold", fold, population, _population_rows(evaluation.predictions, fold, population),
                epoch_by_fold[fold])
        for fold in folds for population in _populations()
    ]
    oof_rows = [
        _metric("oof", None, population,
                _population_rows(evaluation.predictions, None, population), None)
        for population in _populations()
    ]
    return [*fold_rows, *oof_rows]


def validate_dense_baseline_artifacts(
    output_dir: Path, samples: list[FeatureSample], fold_count: int,
    expected_epochs: dict[int, int] | None = None,
) -> None:
    """Validate public artifacts; for example, residual arithmetic is machine-checked."""
    prediction_rows = _read_csv(output_dir / "predictions.csv", PREDICTION_COLUMNS)
    metric_rows = _read_csv(output_dir / "fold_metrics.csv", METRIC_COLUMNS)
    predictions = _validated_predictions(prediction_rows, samples, fold_count)
    actual_epochs = _validate_metric_rows(metric_rows, tuple(predictions), fold_count)
    if expected_epochs is not None and actual_epochs != expected_epochs:
        raise ValueError(
            f"dense metric epochs were {actual_epochs!r}; expected audit epochs "
            f"{expected_epochs!r}"
        )


def _validated_predictions(
    rows: list[dict[str, str]], samples: list[FeatureSample], fold_count: int,
) -> list[DenseOofPrediction]:
    expected_by_name = {sample.file_name: sample for sample in samples}
    names = [row["file_name"] for row in rows]
    if names != sorted(expected_by_name):
        raise ValueError(
            f"dense prediction order/names were {names!r}; expected {sorted(expected_by_name)!r}"
        )
    return [_validated_prediction(row, expected_by_name[row["file_name"]], fold_count)
            for row in rows]


def _validated_prediction(
    row: dict[str, str], sample: FeatureSample, fold_count: int,
) -> DenseOofPrediction:
    predicted = _finite_number(row, "predicted_weight_kg")
    candidate = DenseOofPrediction(
        sample.file_name, sample.fold, sample.weight_category, sample.weight_kg, predicted,
    )
    expected = _prediction_row(candidate)
    if row != expected or not 1 <= sample.fold <= fold_count:
        raise ValueError(
            f"dense prediction row was {row!r}; expected {expected!r} with fold 1..{fold_count}"
        )
    return candidate


def _validate_metric_rows(
    rows: list[dict[str, str]], predictions: tuple[DenseOofPrediction, ...], fold_count: int,
) -> dict[int, int]:
    keys = [(row["scope"], row["fold"], row["population"]) for row in rows]
    expected_keys = _metric_keys(fold_count)
    if keys != expected_keys:
        raise ValueError(f"dense metric keys were {keys!r}; expected {expected_keys!r}")
    epoch_by_fold = _selected_epochs(rows, fold_count)
    audits = tuple(DenseFoldAudit(fold, (), (), (), (), epochs)
                   for fold, epochs in epoch_by_fold.items())
    expected = [_metric_row(metric) for metric in dense_metrics(
        DenseBaselineEvaluation(predictions, audits)
    )]
    if rows != expected:
        raise ValueError(f"dense metric rows were {rows!r}; expected recomputed rows {expected!r}")
    return epoch_by_fold


def _selected_epochs(rows: list[dict[str, str]], fold_count: int) -> dict[int, int]:
    epochs: dict[int, int] = {}
    for fold in range(1, fold_count + 1):
        values = {row["selected_epochs"] for row in rows if row["fold"] == str(fold)}
        if len(values) != 1:
            raise ValueError(f"selected epochs for fold {fold} were {values!r}; expected one value")
        epochs[fold] = _bounded_epoch(values.pop())
    oof_epochs = {row["selected_epochs"] for row in rows if row["scope"] == "oof"}
    if oof_epochs != {""}:
        raise ValueError(f"OOF selected epochs were {oof_epochs!r}; expected only empty text")
    return epochs


def _metric_keys(fold_count: int) -> list[tuple[str, str, str]]:
    fold_keys = [("fold", str(fold), population)
                 for fold in range(1, fold_count + 1) for population in _populations()]
    return [*fold_keys, *(("oof", "", population) for population in _populations())]


def _population_rows(
    predictions: tuple[DenseOofPrediction, ...], fold: int | None, population: Population,
) -> list[DenseOofPrediction]:
    return [row for row in predictions if (fold is None or row.fold == fold)
            and (population == "all" or row.weight_category == population)]


def _metric(
    scope: MetricScope, fold: int | None, population: Population,
    rows: list[DenseOofPrediction], selected_epochs: int | None,
) -> DenseMetric:
    if not rows:
        raise ValueError(
            f"metric population was {population!r} in {scope} fold {fold!r}; "
            "expected at least one OOF prediction"
        )
    observed, predicted = _metric_arrays(rows)
    residuals = predicted - observed
    r2 = _r2(observed, residuals)
    return DenseMetric(scope, fold, population, len(rows), selected_epochs,
                       float(np.mean(np.abs(residuals))),
                       float(np.sqrt(np.mean(np.square(residuals)))),
                       float(np.mean(residuals)), r2)


def _metric_arrays(
    rows: list[DenseOofPrediction],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    observed: NDArray[np.float64] = np.asarray(
        [row.observed_weight_kg for row in rows], dtype=np.float64,
    )
    predicted: NDArray[np.float64] = np.asarray(
        [row.predicted_weight_kg for row in rows], dtype=np.float64,
    )
    return observed, predicted


def _r2(
    observed: NDArray[np.float64], residuals: NDArray[np.float64],
) -> float | None:
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    if len(observed) < 2 or denominator == 0.0:
        return None
    return 1.0 - float(np.sum(np.square(residuals))) / denominator


def _prediction_row(prediction: DenseOofPrediction) -> dict[str, str]:
    residual = prediction.predicted_weight_kg - prediction.observed_weight_kg
    return {
        "model_config": MODEL_CONFIG, "approach": APPROACH,
        "file_name": prediction.file_name, "fold": str(prediction.fold),
        "weight_category": prediction.weight_category,
        "observed_weight_kg": format_csv_number(prediction.observed_weight_kg),
        "predicted_weight_kg": format_csv_number(prediction.predicted_weight_kg),
        "residual_kg": format_csv_number(residual),
        "absolute_error_kg": format_csv_number(abs(residual)),
    }


def _metric_row(metric: DenseMetric) -> dict[str, str]:
    return {
        "model_config": MODEL_CONFIG, "approach": APPROACH, "scope": metric.scope,
        "fold": "" if metric.fold is None else str(metric.fold),
        "population": metric.population, "n": str(metric.n),
        "selected_epochs": "" if metric.selected_epochs is None else str(metric.selected_epochs),
        "mae_kg": format_csv_number(metric.mae_kg),
        "rmse_kg": format_csv_number(metric.rmse_kg),
        "bias_kg": format_csv_number(metric.bias_kg),
        "r2": "" if metric.r2 is None else format_csv_number(metric.r2),
    }


def _read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        actual, rows = list(reader.fieldnames or []), list(reader)
    if actual != columns:
        raise ValueError(f"dense CSV columns were {actual!r}; expected exactly {columns!r}")
    if any(not math.isfinite(float(row[field])) for row in rows
           for field in _required_numeric_fields(path.name)):
        raise ValueError(f"dense CSV at {path} contained non-finite values; expected finite numbers")
    return rows


def _finite_number(row: dict[str, str], field: str) -> float:
    value = row.get(field)
    try:
        parsed = float(value or "")
    except ValueError as error:
        raise ValueError(f"{field} was {value!r}; expected finite numeric text") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} was {value!r}; expected finite numeric text")
    return parsed


def _bounded_epoch(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(
            f"selected_epochs was {value!r}; expected integer text in "
            f"1..{DENSE_BASELINE_RECIPE.max_epochs}"
        ) from error
    if not 1 <= parsed <= DENSE_BASELINE_RECIPE.max_epochs:
        raise ValueError(
            f"selected_epochs was {value!r}; expected 1..{DENSE_BASELINE_RECIPE.max_epochs}"
        )
    return parsed


def _required_numeric_fields(file_name: str) -> tuple[str, ...]:
    if file_name == "predictions.csv":
        return ("observed_weight_kg", "predicted_weight_kg", "residual_kg", "absolute_error_kg")
    return ("mae_kg", "rmse_kg", "bias_kg")


def _populations() -> tuple[Population, ...]:
    return ("all", "B1", "B10")
