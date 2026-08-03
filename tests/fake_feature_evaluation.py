from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.feature_evaluation import (
    FeatureBaseline,
    FeaturePredictor,
    PredictionPartition,
    TrainingPartition,
)


class UnavailableCudaRuntime:
    """Report unavailable CUDA without creating a model or allocating tensors."""

    def __init__(self) -> None:
        """Track preflight checks; for example, no model operation should follow failure."""
        self.availability_checks = 0
        self.model_operations = 0

    def cuda_available(self) -> bool:
        self.availability_checks += 1
        available = False
        return available


@dataclass(frozen=True)
class RecordedFit:
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]


class FirstColumnPredictor(FeaturePredictor):
    def __init__(self, owner: "RecordingFeatureBaseline") -> None:
        """Retain the recording owner; for example, predictions append observed partitions."""
        self._owner = owner
        self._column_index = 0

    def predict(self, partition: PredictionPartition) -> NDArray[np.float64]:
        self._owner.predicted_partitions.append(partition)
        predictions = partition.values[:, self._column_index].copy()
        return predictions


class RecordingFeatureBaseline(FeatureBaseline):
    """Record adapter boundaries while predicting from the first selected column."""

    def __init__(self, name: str = "random_forest") -> None:
        self.name = name
        self.fit_calls: list[RecordedFit] = []
        self.predicted_partitions: list[PredictionPartition] = []

    def fit(
        self, partition: TrainingPartition, feature_names: tuple[str, ...]
    ) -> FeaturePredictor:
        self.fit_calls.append(RecordedFit(partition.sample_ids, feature_names))
        return FirstColumnPredictor(self)
