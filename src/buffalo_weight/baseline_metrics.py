"""Regression metrics for one baseline configuration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.baseline_types import BaselinePrediction


@dataclass(frozen=True)
class MetricSummary:
    n: int
    mae_kg: float
    rmse_kg: float
    bias_kg: float
    r2: float


def summarize_predictions(predictions: list[BaselinePrediction]) -> MetricSummary:
    """Calculate one direct summary; for example, OOF metrics pool every mask."""
    observed = np.asarray([row.observed_weight_kg for row in predictions], dtype=np.float64)
    predicted = np.asarray([row.predicted_weight_kg for row in predictions], dtype=np.float64)
    residuals = predicted - observed
    return MetricSummary(
        len(predictions), float(np.mean(np.abs(residuals))),
        float(np.sqrt(np.mean(np.square(residuals)))), float(np.mean(residuals)),
        _r2(observed, residuals),
    )


def fold_summaries(
    predictions: list[BaselinePrediction],
) -> list[tuple[int, MetricSummary]]:
    """Summarize canonical folds; for example, output order is 1 through 5."""
    return [
        (fold, summarize_predictions([row for row in predictions if row.fold == fold]))
        for fold in sorted({row.fold for row in predictions})
    ]


def grouped_summaries(
    predictions: list[BaselinePrediction],
) -> list[tuple[str, MetricSummary]]:
    """Summarize full OOF, B1 and B10; for example, ``all`` is pooled, not fold-averaged."""
    populations = (
        ("all", predictions),
        ("B1", [row for row in predictions if row.weight_category == "B1"]),
        ("B10", [row for row in predictions if row.weight_category == "B10"]),
    )
    return [(name, summarize_predictions(rows)) for name, rows in populations]


def _r2(observed: NDArray[np.float64], residuals: NDArray[np.float64]) -> float:
    centered = observed - np.mean(observed)
    denominator = float(np.sum(np.square(centered)))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - float(np.sum(np.square(residuals))) / denominator
