"""Typed records for reconstructible baseline evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BaselineConfiguration = Literal["random_forest_baseline", "training_mean_reference"]
EvaluationRole = Literal["candidate", "reference"]


@dataclass(frozen=True)
class BaselinePrediction:
    file_name: str
    weight_category: str
    fold: int
    observed_weight_kg: float
    predicted_weight_kg: float
    configuration: BaselineConfiguration
    evaluation_role: EvaluationRole

    @property
    def residual_kg(self) -> float:
        """Return signed error; for example, positive values mean overestimation."""
        residual = self.predicted_weight_kg - self.observed_weight_kg
        return residual

    @property
    def absolute_error_kg(self) -> float:
        """Return error magnitude; for example, a -2 kg residual becomes 2 kg."""
        magnitude = abs(self.residual_kg)
        return magnitude
