"""Shared records for controlled comparison of baseline configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


EvaluationRole = Literal["candidate", "reference"]
MetricScope = Literal["fold", "oof"]
MetricPopulation = Literal["all", "B1", "B10"]


@dataclass(frozen=True)
class ComparisonPrediction:
    configuration: str
    approach: str
    evaluation_role: EvaluationRole
    file_name: str
    weight_category: str
    fold: int
    observed_weight_kg: float
    predicted_weight_kg: float

    @property
    def residual_kg(self) -> float:
        """Return signed error; for example, ``2`` denotes overestimation by 2 kg."""
        predicted = self.predicted_weight_kg
        observed = self.observed_weight_kg
        return predicted - observed


@dataclass(frozen=True)
class ComparisonMetric:
    configuration: str
    approach: str
    evaluation_role: EvaluationRole
    scope: MetricScope
    fold: int | None
    population: MetricPopulation
    n: int
    mae_kg: float
    rmse_kg: float | None
    bias_kg: float
    r2: float | None
