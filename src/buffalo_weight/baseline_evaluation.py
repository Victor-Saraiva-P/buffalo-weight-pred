"""Leak-free outer-fold evaluation for report baselines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.baseline_types import (
    BaselineConfiguration,
    BaselinePrediction,
    EvaluationRole,
)
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
    for training, held_out in _outer_fold_partitions(samples):
        predictor = baseline.fit(_training_partition(training, feature_names), feature_names)
        predicted_weights = predictor.predict(_prediction_partition(held_out, feature_names))
        predictions.extend(_prediction_rows(held_out, predicted_weights))
    return sorted(predictions, key=lambda row: row.file_name)


def evaluate_training_mean_reference(samples: list[FeatureSample]) -> list[BaselinePrediction]:
    """Predict each fold from its training mean; for example, fold 1 never informs its mean."""
    predictions: list[BaselinePrediction] = []
    for training, held_out in _outer_fold_partitions(samples):
        training_weights = [sample.weight_kg for sample in training]
        fold_mean = float(np.mean(np.asarray(training_weights, dtype=np.float64)))
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
    feature_matrix_rows = [
        [sample.feature_values[name] for name in feature_names] for sample in samples
    ]
    return np.asarray(feature_matrix_rows, dtype=np.float64)


def _outer_fold_partitions(
    samples: list[FeatureSample],
) -> list[tuple[list[FeatureSample], list[FeatureSample]]]:
    folds = sorted({sample.fold for sample in samples})
    return [
        ([sample for sample in samples if sample.fold != fold],
         [sample for sample in samples if sample.fold == fold])
        for fold in folds
    ]


def _prediction_rows(
    samples: list[FeatureSample], predicted_weights: NDArray[np.float64],
) -> list[BaselinePrediction]:
    if predicted_weights.shape != (len(samples),):
        raise ValueError(
            f"prediction shape was {predicted_weights.shape!r}; expected {(len(samples),)!r}"
        )
    return [
        _baseline_prediction(sample, float(prediction), "random_forest_baseline", "candidate")
        for sample, prediction in zip(samples, predicted_weights, strict=True)
    ]


def _reference_rows(
    samples: list[FeatureSample], prediction: float,
) -> list[BaselinePrediction]:
    return [
        _baseline_prediction(sample, prediction, "training_mean_reference", "reference")
        for sample in samples
    ]


def _baseline_prediction(
    sample: FeatureSample, predicted_weight_kg: float,
    configuration: BaselineConfiguration, role: EvaluationRole,
) -> BaselinePrediction:
    return BaselinePrediction(
        sample.file_name, sample.weight_category, sample.fold, sample.weight_kg,
        predicted_weight_kg, configuration, role,
    )
