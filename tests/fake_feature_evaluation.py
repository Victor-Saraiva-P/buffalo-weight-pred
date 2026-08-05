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
from buffalo_weight.feature_selection_types import FeatureBaselineName


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


@dataclass(frozen=True)
class RecordedPrediction:
    training_ids: tuple[str, ...]
    prediction_ids: tuple[str, ...]


class FirstColumnPredictor(FeaturePredictor):
    def __init__(
        self, owner: "RecordingFeatureBaseline", training_ids: tuple[str, ...]
    ) -> None:
        """Retain the recording owner; for example, predictions append observed partitions."""
        self._owner = owner
        self._training_ids = training_ids
        self._column_index = 0

    def predict(self, partition: PredictionPartition) -> NDArray[np.float64]:
        self._owner.predicted_partitions.append(partition)
        self._owner.prediction_calls.append(
            RecordedPrediction(self._training_ids, partition.sample_ids)
        )
        predictions = partition.values[:, self._column_index].copy()
        return predictions


class RecordingFeatureBaseline(FeatureBaseline):
    """Record adapter boundaries while predicting from the first selected column."""

    def __init__(self, name: FeatureBaselineName = "random_forest") -> None:
        self.name = name
        self.fit_calls: list[RecordedFit] = []
        self.predicted_partitions: list[PredictionPartition] = []
        self.prediction_calls: list[RecordedPrediction] = []

    def fit(
        self, partition: TrainingPartition, feature_names: tuple[str, ...]
    ) -> FeaturePredictor:
        self.fit_calls.append(RecordedFit(partition.sample_ids, feature_names))
        return FirstColumnPredictor(self, partition.sample_ids)


class ConstantFeaturePredictor(FeaturePredictor):
    """Return a known literal so artifact metrics have an independent oracle."""

    def predict(self, partition: PredictionPartition) -> NDArray[np.float64]:
        return np.full(len(partition.sample_ids), 100.0, dtype=np.float64)


class ConstantFeatureBaseline(FeatureBaseline):
    """Fit a deterministic boundary whose predictions are always 100 kg."""

    name: FeatureBaselineName = "random_forest"

    def fit(
        self, partition: TrainingPartition, feature_names: tuple[str, ...]
    ) -> FeaturePredictor:
        del partition, feature_names
        return ConstantFeaturePredictor()
