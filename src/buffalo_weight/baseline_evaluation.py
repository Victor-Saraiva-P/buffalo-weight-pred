"""Leak-free outer-fold evaluation for report baselines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.baseline_types import BaselinePrediction
from buffalo_weight.feature_evaluation import (
    FeatureBaseline,
    FeatureSample,
    PredictionPartition,
    TrainingPartition,
)


def evaluate_random_forest_oof(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
    baseline: FeatureBaseline,
) -> list[BaselinePrediction]:
    """Predict every outer fold; for example, held-out rows never enter ``fit``."""
    predictions: list[BaselinePrediction] = []
    for fold in sorted({sample.fold for sample in samples}):
        training = [sample for sample in samples if sample.fold != fold]
        held_out = [sample for sample in samples if sample.fold == fold]
        predictor = baseline.fit(_training_partition(training, feature_names), feature_names)
        values = predictor.predict(_prediction_partition(held_out, feature_names))
        predictions.extend(_prediction_rows(held_out, values))
    return sorted(predictions, key=lambda row: row.file_name)


def evaluate_training_mean_reference(samples: list[FeatureSample]) -> list[BaselinePrediction]:
    """Predict each fold from its training mean; for example, fold 1 never informs its mean."""
    predictions: list[BaselinePrediction] = []
    for fold in sorted({sample.fold for sample in samples}):
        training_weights = [sample.weight_kg for sample in samples if sample.fold != fold]
        fold_mean = float(np.mean(np.asarray(training_weights, dtype=np.float64)))
        held_out = [sample for sample in samples if sample.fold == fold]
        predictions.extend(_reference_rows(held_out, fold_mean))
    return sorted(predictions, key=lambda row: row.file_name)


def _training_partition(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
) -> TrainingPartition:
    return TrainingPartition(
        _feature_matrix(samples, feature_names),
        np.asarray([sample.weight_kg for sample in samples], dtype=np.float64),
        tuple(sample.file_name for sample in samples),
        tuple(sample.weight_category for sample in samples),
    )


def _prediction_partition(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
) -> PredictionPartition:
    return PredictionPartition(
        _feature_matrix(samples, feature_names),
        tuple(sample.file_name for sample in samples),
    )


def _feature_matrix(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
) -> NDArray[np.float64]:
    values = [[sample.feature_values[name] for name in feature_names] for sample in samples]
    return np.asarray(values, dtype=np.float64)


def _prediction_rows(
    samples: list[FeatureSample], predictions: NDArray[np.float64],
) -> list[BaselinePrediction]:
    if predictions.shape != (len(samples),):
        raise ValueError(
            f"prediction shape was {predictions.shape!r}; expected {(len(samples),)!r}"
        )
    return [
        BaselinePrediction(sample.file_name, sample.weight_category, sample.fold,
                           sample.weight_kg, float(prediction),
                           "random_forest_baseline", "candidate")
        for sample, prediction in zip(samples, predictions, strict=True)
    ]


def _reference_rows(
    samples: list[FeatureSample], prediction: float,
) -> list[BaselinePrediction]:
    return [
        BaselinePrediction(sample.file_name, sample.weight_category, sample.fold,
                           sample.weight_kg, prediction,
                           "training_mean_reference", "reference")
        for sample in samples
    ]
