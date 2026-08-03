"""Cross-fold orchestration for comparative feature evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.feature_selection_rules import classify_mae_delta, permutation_seed


@dataclass(frozen=True)
class FeatureSample:
    file_name: str
    fold: int
    weight_category: str
    weight_kg: float
    feature_values: dict[str, float]


@dataclass(frozen=True)
class RemovalGroup:
    name: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class TrainingPartition:
    values: NDArray[np.float64]
    targets_kg: NDArray[np.float64]
    sample_ids: tuple[str, ...]
    strata: tuple[str, ...]


@dataclass(frozen=True)
class PredictionPartition:
    values: NDArray[np.float64]
    sample_ids: tuple[str, ...]


class FeaturePredictor(Protocol):
    def predict(self, partition: PredictionPartition) -> NDArray[np.float64]: ...


class FeatureBaseline(Protocol):
    name: str

    def fit(
        self, partition: TrainingPartition, feature_names: tuple[str, ...]
    ) -> FeaturePredictor: ...


@dataclass(frozen=True)
class FeatureEvidence:
    experiment: str
    baseline: str
    target: str
    scope: str
    fold: int | None
    repetition: int | None
    permutation_seed: int | None
    n: int
    reference_mae_kg: float | None
    result_mae_kg: float
    delta_mae_kg: float | None
    effect: str | None


@dataclass(frozen=True)
class _PredictionBatch:
    experiment: str
    baseline: str
    target: str
    fold: int
    repetition: int | None
    seed: int | None
    targets: NDArray[np.float64]
    reference: NDArray[np.float64] | None
    predictions: NDArray[np.float64]


def evaluate_feature_evidence(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
    removal_groups: tuple[RemovalGroup, ...], baselines: tuple[FeatureBaseline, ...],
    permutation_count: int, split_seed: int,
) -> list[FeatureEvidence]:
    """Evaluate held-out evidence; for example, pass both frozen baseline adapters."""
    batches: list[_PredictionBatch] = []
    folds = sorted({sample.fold for sample in samples})
    for baseline in baselines:
        for fold in folds:
            batches.extend(_evaluate_fold(samples, fold, feature_names, removal_groups, baseline,
                                          permutation_count, split_seed))
    return _evidence_rows(batches)


def _evaluate_fold(
    samples: list[FeatureSample], fold: int, feature_names: tuple[str, ...],
    removal_groups: tuple[RemovalGroup, ...], baseline: FeatureBaseline,
    permutation_count: int, split_seed: int,
) -> list[_PredictionBatch]:
    train = [sample for sample in samples if sample.fold != fold]
    held_out = [sample for sample in samples if sample.fold == fold]
    full_predictor = baseline.fit(_training_partition(train, feature_names), feature_names)
    full_predictions = full_predictor.predict(_prediction_partition(held_out, feature_names))
    batches = _isolated_batches(train, held_out, feature_names, baseline, fold)
    batches.extend(_removal_batches(train, held_out, feature_names, removal_groups,
                                    baseline, fold, full_predictions))
    batches.extend(_permutation_batches(held_out, feature_names, baseline.name, fold,
                                        full_predictor, full_predictions, permutation_count, split_seed))
    return batches


def _isolated_batches(
    train: list[FeatureSample], held_out: list[FeatureSample],
    feature_names: tuple[str, ...], baseline: FeatureBaseline, fold: int,
) -> list[_PredictionBatch]:
    batches = []
    for feature in feature_names:
        selected = (feature,)
        predictor = baseline.fit(_training_partition(train, selected), selected)
        predictions = predictor.predict(_prediction_partition(held_out, selected))
        batches.append(_batch("isolated", baseline.name, feature, fold, held_out, None, predictions))
    return batches


def _removal_batches(
    train: list[FeatureSample], held_out: list[FeatureSample], feature_names: tuple[str, ...],
    groups: tuple[RemovalGroup, ...], baseline: FeatureBaseline, fold: int,
    reference: NDArray[np.float64],
) -> list[_PredictionBatch]:
    removals: list[tuple[str, tuple[str, ...]]] = [
        (name, (name,)) for name in feature_names
    ]
    removals.extend((group.name, group.features) for group in groups)
    viable = [(target, removed) for target, removed in removals
              if any(name not in removed for name in feature_names)]
    return [_removal_batch(train, held_out, feature_names, baseline, fold, reference, target, removed)
            for target, removed in viable]


def _removal_batch(
    train: list[FeatureSample], held_out: list[FeatureSample], feature_names: tuple[str, ...],
    baseline: FeatureBaseline, fold: int, reference: NDArray[np.float64],
    target: str, removed: tuple[str, ...],
) -> _PredictionBatch:
    selected = tuple(name for name in feature_names if name not in removed)
    predictor = baseline.fit(_training_partition(train, selected), selected)
    predictions = predictor.predict(_prediction_partition(held_out, selected))
    return _batch("removal", baseline.name, target, fold, held_out, reference, predictions)


def _permutation_batches(
    held_out: list[FeatureSample], feature_names: tuple[str, ...], baseline_name: str,
    fold: int, predictor: FeaturePredictor, reference: NDArray[np.float64],
    permutation_count: int, split_seed: int,
) -> list[_PredictionBatch]:
    batches = []
    for feature_index, feature in enumerate(feature_names):
        for repetition in range(permutation_count):
            seed = permutation_seed(split_seed, fold, feature, repetition)
            partition = _permuted_partition(held_out, feature_names, feature_index, seed)
            predictions = predictor.predict(partition)
            batches.append(_batch("permutation", baseline_name, feature, fold, held_out,
                                  reference, predictions, repetition, seed))
    return batches


def _training_partition(
    samples: list[FeatureSample], feature_names: tuple[str, ...]
) -> TrainingPartition:
    return TrainingPartition(_matrix(samples, feature_names), _targets(samples),
                             tuple(sample.file_name for sample in samples),
                             tuple(sample.weight_category for sample in samples))


def _prediction_partition(
    samples: list[FeatureSample], feature_names: tuple[str, ...]
) -> PredictionPartition:
    return PredictionPartition(_matrix(samples, feature_names),
                               tuple(sample.file_name for sample in samples))


def _permuted_partition(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
    feature_index: int, seed: int,
) -> PredictionPartition:
    partition = _prediction_partition(samples, feature_names)
    values = partition.values.copy()
    values[:, feature_index] = np.random.default_rng(seed).permutation(values[:, feature_index])
    return PredictionPartition(values, partition.sample_ids)


def _matrix(
    samples: list[FeatureSample], feature_names: tuple[str, ...]
) -> NDArray[np.float64]:
    return np.asarray([[sample.feature_values[name] for name in feature_names]
                       for sample in samples], dtype=np.float64)


def _targets(samples: list[FeatureSample]) -> NDArray[np.float64]:
    return np.asarray([sample.weight_kg for sample in samples], dtype=np.float64)


def _batch(
    experiment: str, baseline: str, target: str, fold: int,
    samples: list[FeatureSample], reference: NDArray[np.float64] | None,
    predictions: NDArray[np.float64], repetition: int | None = None,
    seed: int | None = None,
) -> _PredictionBatch:
    return _PredictionBatch(experiment, baseline, target, fold, repetition, seed,
                            _targets(samples), reference, predictions)


def _evidence_rows(batches: list[_PredictionBatch]) -> list[FeatureEvidence]:
    fold_rows = [_evidence_row(batch, "fold", batch.fold) for batch in batches]
    grouped: dict[tuple[str, str, str, int | None], list[_PredictionBatch]] = {}
    for batch in batches:
        key = (batch.experiment, batch.baseline, batch.target, batch.repetition)
        grouped.setdefault(key, []).append(batch)
    oof_rows = [_oof_evidence(group) for group in grouped.values()]
    return sorted([*fold_rows, *oof_rows], key=_evidence_sort_key)


def _oof_evidence(batches: list[_PredictionBatch]) -> FeatureEvidence:
    first = batches[0]
    combined = _PredictionBatch(first.experiment, first.baseline, first.target, 0,
                                first.repetition, None,
                                np.concatenate([batch.targets for batch in batches]),
                                _combined_reference(batches),
                                np.concatenate([batch.predictions for batch in batches]))
    return _evidence_row(combined, "oof", None)


def _combined_reference(batches: list[_PredictionBatch]) -> NDArray[np.float64] | None:
    if batches[0].reference is None:
        return None
    return np.concatenate([batch.reference for batch in batches if batch.reference is not None])


def _evidence_row(batch: _PredictionBatch, scope: str, fold: int | None) -> FeatureEvidence:
    result_mae = _mae(batch.targets, batch.predictions)
    reference_mae = None if batch.reference is None else _mae(batch.targets, batch.reference)
    delta = None if reference_mae is None else result_mae - reference_mae
    effect = None if delta is None else classify_mae_delta(delta)
    return FeatureEvidence(batch.experiment, batch.baseline, batch.target, scope, fold,
                           batch.repetition, batch.seed, len(batch.targets), reference_mae,
                           result_mae, delta, effect)


def _mae(targets: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
    return float(np.mean(np.abs(targets - predictions)))


def _evidence_sort_key(row: FeatureEvidence) -> tuple[object, ...]:
    scope_rank = 0 if row.scope == "fold" else 1
    return (row.experiment, row.baseline, row.target, row.repetition or 0,
            scope_rank, row.fold or 0)
