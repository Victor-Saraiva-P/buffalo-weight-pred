"""OOF evaluation for the frozen Rede Densa por Feições baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.feature_baselines import DenseFeatureBaseline
from buffalo_weight.feature_evaluation import (
    FeatureSample,
    PredictionPartition,
    TrainingPartition,
)


@dataclass(frozen=True)
class DenseOofPrediction:
    file_name: str
    fold: int
    weight_category: str
    observed_weight_kg: float
    predicted_weight_kg: float


@dataclass(frozen=True)
class DenseFoldAudit:
    fold: int
    selection_ids: tuple[str, ...]
    stopping_ids: tuple[str, ...]
    retrain_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    selected_epochs: int


@dataclass(frozen=True)
class DenseBaselineEvaluation:
    predictions: tuple[DenseOofPrediction, ...]
    fold_audits: tuple[DenseFoldAudit, ...]


class DenseBaselineRunner(Protocol):
    """Evaluation seam; for example, CLI tests inject a deterministic runner."""

    def evaluate(
        self, samples: list[FeatureSample], feature_names: tuple[str, ...],
    ) -> DenseBaselineEvaluation:
        """Evaluate external folds; for example, each sample receives one OOF prediction."""
        ...


class ScientificDenseBaselineRunner:
    """Run the frozen CUDA baseline; for example, production CLI uses this runner."""

    def __init__(self, baseline: DenseFeatureBaseline | None = None) -> None:
        """Inject the model seam; for example, tests replace CUDA training with a recorder."""
        self._baseline = baseline

    def evaluate(
        self, samples: list[FeatureSample], feature_names: tuple[str, ...],
    ) -> DenseBaselineEvaluation:
        """Produce OOF predictions; for example, the external fold never enters fitting."""
        baseline = self._baseline or DenseFeatureBaseline()
        predictions: list[DenseOofPrediction] = []
        audits: list[DenseFoldAudit] = []
        for fold in sorted({sample.fold for sample in samples}):
            fold_predictions, fold_audit = _evaluate_fold(
                samples, feature_names, fold, baseline,
            )
            predictions.extend(fold_predictions)
            audits.append(fold_audit)
        return DenseBaselineEvaluation(tuple(predictions), tuple(audits))


def _evaluate_fold(
    samples: list[FeatureSample], feature_names: tuple[str, ...], fold: int,
    baseline: DenseFeatureBaseline,
) -> tuple[list[DenseOofPrediction], DenseFoldAudit]:
    train = [sample for sample in samples if sample.fold != fold]
    held_out = [sample for sample in samples if sample.fold == fold]
    predictor = baseline.fit(_training_partition(train, feature_names), feature_names)
    values = _feature_matrix(held_out, feature_names)
    held_out_ids = tuple(sample.file_name for sample in held_out)
    predicted = predictor.predict(PredictionPartition(values, held_out_ids))
    if len(predicted) != len(held_out):
        raise ValueError(
            f"dense predictions were {len(predicted)} for fold {fold}; "
            f"expected {len(held_out)} held-out values"
        )
    records = [_prediction_record(sample, value) for sample, value in zip(held_out, predicted)]
    return records, _fold_audit(fold, held_out_ids, baseline)


def _training_partition(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
) -> TrainingPartition:
    return TrainingPartition(
        _feature_matrix(samples, feature_names),
        np.asarray([sample.weight_kg for sample in samples], dtype=np.float64),
        tuple(sample.file_name for sample in samples),
        tuple(sample.weight_category for sample in samples),
    )


def _feature_matrix(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
) -> NDArray[np.float64]:
    values = [[sample.feature_values[name] for name in feature_names] for sample in samples]
    return np.asarray(values, dtype=np.float64)


def _prediction_record(sample: FeatureSample, predicted: np.float64) -> DenseOofPrediction:
    return DenseOofPrediction(
        sample.file_name, sample.fold, sample.weight_category, sample.weight_kg,
        float(predicted),
    )


def _fold_audit(
    fold: int, held_out_ids: tuple[str, ...], baseline: DenseFeatureBaseline,
) -> DenseFoldAudit:
    audit = baseline.training_audits[-1]
    return DenseFoldAudit(
        fold, audit.selection_ids, audit.stopping_ids, audit.retrain_ids,
        held_out_ids, audit.selected_epochs,
    )
