"""Typed records for reconstructible baseline evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BaselineConfiguration = Literal["random_forest_baseline", "training_mean_reference"]
EvaluationRole = Literal["candidate", "reference"]
BaselineStatus = Literal["absent", "blocked", "obsolete", "rebuilt", "reusable"]
BASELINE_VALIDATIONS = [
    "schemas", "ordering", "sha256", "oof_uniqueness", "outer_fold_isolation",
]


@dataclass(frozen=True)
class BaselineDefinition:
    configuration: BaselineConfiguration
    evaluation_role: EvaluationRole
    consumes_confirmed_features: bool
    dependencies: tuple[str, ...]


BASELINE_DEFINITIONS = (
    BaselineDefinition(
        "random_forest_baseline", "candidate", True, ("numpy", "scikit-learn"),
    ),
    BaselineDefinition(
        "training_mean_reference", "reference", False, ("numpy",),
    ),
)


def baseline_definition(configuration: BaselineConfiguration) -> BaselineDefinition:
    """Return frozen metadata; for example, the mean reference consumes no features."""
    matching = [item for item in BASELINE_DEFINITIONS if item.configuration == configuration]
    if len(matching) != 1:
        raise ValueError(
            f"baseline configuration was {configuration!r}; expected one frozen definition"
        )
    return matching[0]


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
