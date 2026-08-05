"""Outer-fold orchestration for the frozen ResNet-18 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass(frozen=True)
class ResNetSample:
    file_name: str
    mask_path: Path
    weight_category: str
    fold: int
    weight_kg: float


@dataclass(frozen=True)
class ResNetOofPrediction:
    file_name: str
    fold: int
    weight_category: str
    weight_kg: float
    prediction_kg: float


class ResNetBaselinePredictor(Protocol):
    """Inference seam; for example, outer-fold evaluation supplies only reserved rows."""

    def predict(self, samples: tuple[ResNetSample, ...]) -> NDArray[np.float64]:
        """Predict kilograms; for example, one value is returned for each reserved mask."""
        ...


class ResNetTrainingAdapter(Protocol):
    """Training seam; for example, tests record selection and refit partitions."""

    def select_epoch_count(
        self, training: tuple[ResNetSample, ...], validation: tuple[ResNetSample, ...]
    ) -> int:
        """Choose only partial-fit epochs; for example, inner validation returns seven."""
        ...

    def fit_epochs(
        self, training: tuple[ResNetSample, ...], partial_epochs: int
    ) -> ResNetBaselinePredictor:
        """Refit from official weights; for example, use every permitted outer-train row."""
        ...


class ResNetBaselineEvaluator:
    """Produce isolated Predições OOF; for example, evaluate all canonical folds."""

    def __init__(self, adapter: ResNetTrainingAdapter, inner_seed: int = 43) -> None:
        self._adapter = adapter
        self._inner_seed = inner_seed

    def evaluate(self, samples: tuple[ResNetSample, ...]) -> list[ResNetOofPrediction]:
        """Evaluate every fold; for example, no reserved mask reaches selection or refit."""
        predictions: list[ResNetOofPrediction] = []
        for fold in sorted({sample.fold for sample in samples}):
            predictions.extend(self._evaluate_fold(samples, fold))
        return sorted(predictions, key=lambda row: row.file_name)

    def _evaluate_fold(
        self, samples: tuple[ResNetSample, ...], fold: int
    ) -> list[ResNetOofPrediction]:
        permitted = tuple(sample for sample in samples if sample.fold != fold)
        reserved = tuple(sample for sample in samples if sample.fold == fold)
        training, validation = _inner_partition(permitted, self._inner_seed)
        partial_epochs = self._adapter.select_epoch_count(training, validation)
        predictor = self._adapter.fit_epochs(permitted, partial_epochs)
        values = predictor.predict(reserved)
        return [_prediction(sample, float(value)) for sample, value in zip(reserved, values)]


def _inner_partition(
    samples: tuple[ResNetSample, ...], seed: int
) -> tuple[tuple[ResNetSample, ...], tuple[ResNetSample, ...]]:
    categories = [sample.weight_category for sample in samples]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    training, validation = next(splitter.split(np.zeros(len(samples)), categories))
    return _selected_samples(samples, training), _selected_samples(samples, validation)


def _selected_samples(
    samples: tuple[ResNetSample, ...], indices: NDArray[np.int64]
) -> tuple[ResNetSample, ...]:
    return tuple(samples[int(index)] for index in indices)


def _prediction(sample: ResNetSample, value: float) -> ResNetOofPrediction:
    return ResNetOofPrediction(
        sample.file_name, sample.fold, sample.weight_category, sample.weight_kg, value
    )
