"""Direct metrics over normalized OOF baseline predictions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.baseline_comparison_types import (
    ComparisonMetric,
    ComparisonPrediction,
    EvaluationRole,
    MetricPopulation,
    MetricScope,
)


def comparison_metric_rows(
    predictions: list[ComparisonPrediction],
) -> list[ComparisonMetric]:
    """Build tidy metrics; for example, pooled OOF rows use every prediction directly."""
    if not predictions:
        raise ValueError(
            f"comparison predictions were {predictions!r}; expected at least one finite row"
        )
    configuration, approach, role = _single_identity(predictions)
    fold_rows = [
        _summarize(configuration, approach, role, "fold", fold, "all", rows)
        for fold, rows in _predictions_by_fold(predictions)
    ]
    oof_rows = [
        _summarize(configuration, approach, role, "oof", None, population,
                   _population_predictions(predictions, population))
        for population in ("all", "B1", "B10")
    ]
    return [*fold_rows, *oof_rows]


def _single_identity(
    predictions: list[ComparisonPrediction],
) -> tuple[str, str, EvaluationRole]:
    identities = {(row.configuration, row.approach, row.evaluation_role)
                  for row in predictions}
    if len(identities) != 1:
        raise ValueError(f"comparison identities were {identities!r}; expected exactly one")
    return next(iter(identities))


def _predictions_by_fold(
    predictions: list[ComparisonPrediction],
) -> list[tuple[int, list[ComparisonPrediction]]]:
    folds = sorted({row.fold for row in predictions})
    return [(fold, [row for row in predictions if row.fold == fold]) for fold in folds]


def _population_predictions(
    predictions: list[ComparisonPrediction], population: MetricPopulation,
) -> list[ComparisonPrediction]:
    if population == "all":
        return predictions
    selected = [row for row in predictions if row.weight_category == population]
    if not selected:
        raise ValueError(f"comparison population was {population!r}; expected OOF predictions")
    return selected


def _summarize(
    configuration: str, approach: str, role: EvaluationRole, scope: MetricScope,
    fold: int | None, population: MetricPopulation,
    predictions: list[ComparisonPrediction],
) -> ComparisonMetric:
    observed, estimated = _metric_arrays(predictions)
    residuals = estimated - observed
    detailed = population == "all"
    return ComparisonMetric(
        configuration, approach, role, scope, fold, population, len(predictions),
        float(np.mean(np.abs(residuals))),
        float(np.sqrt(np.mean(np.square(residuals)))) if detailed else None,
        float(np.mean(residuals)), _r2(observed, residuals) if detailed else None,
    )


def _metric_arrays(
    predictions: list[ComparisonPrediction],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    observed = np.asarray([row.observed_weight_kg for row in predictions], dtype=np.float64)
    estimated = np.asarray([row.predicted_weight_kg for row in predictions], dtype=np.float64)
    invalid = [(row.file_name, row.observed_weight_kg, row.predicted_weight_kg)
               for row in predictions if not math.isfinite(row.observed_weight_kg)
               or not math.isfinite(row.predicted_weight_kg)]
    if invalid:
        raise ValueError(f"comparison values were {invalid!r}; expected finite weights")
    return observed, estimated


def _r2(observed: NDArray[np.float64], residuals: NDArray[np.float64]) -> float | None:
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    if len(observed) < 2 or denominator == 0.0:
        return None
    return 1.0 - float(np.sum(np.square(residuals))) / denominator
