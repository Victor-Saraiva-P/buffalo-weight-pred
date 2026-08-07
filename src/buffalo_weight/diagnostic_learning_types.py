"""Types and data structures for controlled learning curves slice.

Reference: GitHub Issue #25.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LEARNING_CURVE_CONFIGURATIONS = (
    "random_forest_baseline",
    "dense_baseline",
    "compact_cnn_baseline",
    "resnet18_pretrained_partial",
)

EvaluatedPopulation = Literal["oof"]
ArtifactAction = Literal["reused", "retrained"]


@dataclass(frozen=True)
class LearningPointRecord:
    """Evaluation point for one baseline fold and fraction.

    Example: ``LearningPointRecord("random_forest_baseline", 1, 0.50, 53, "oof", 26, 12.5, -0.4, "retrained")``.
    """

    configuration: str
    fold: int
    fraction: float
    n_train: int
    evaluated_population: EvaluatedPopulation
    n_eval: int
    mae_kg: float
    bias_kg: float
    artifact_action: ArtifactAction


@dataclass(frozen=True)
class LearningCurveSummaryRecord:
    """Summary of learning curve metrics across folds for one configuration and fraction.

    Example: ``LearningCurveSummaryRecord("random_forest_baseline", 0.50, 53.0, 12.5, 0.8, -0.4, 0)``.
    """

    configuration: str
    fraction: float
    mean_n_train: float
    mean_mae_kg: float
    std_mae_kg: float
    mean_bias_kg: float
    reused_points_count: int


@dataclass(frozen=True)
class LearningCurvesSlice:
    """Complete diagnostic slice for controlled learning curves.

    Example: ``LearningCurvesSlice(point_records, summary_records)``.
    """

    point_records: tuple[LearningPointRecord, ...]
    summary_records: tuple[LearningCurveSummaryRecord, ...]
