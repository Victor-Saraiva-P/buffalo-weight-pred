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


@dataclass(frozen=True)
class RecordedFit:
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]


class FirstColumnPredictor(FeaturePredictor):
    def __init__(self, owner: "RecordingFeatureBaseline") -> None:
        self._owner = owner

    def predict(self, partition: PredictionPartition) -> NDArray[np.float64]:
        self._owner.predicted_partitions.append(partition)
        return partition.values[:, 0].copy()


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
