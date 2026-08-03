"""Frozen model adapters used by feature-selection evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit

from buffalo_weight.dense_feature_adapter import (
    DenseFeatureAdapter,
    DenseFeatureNetwork,
    DenseTrainingRecipe,
)
from buffalo_weight.feature_evaluation import (
    FeaturePredictor,
    PredictionPartition,
    TrainingPartition,
)


class SklearnFeaturePredictor:
    """Own scikit-learn inference; for example, RF training returns this predictor."""

    def __init__(self, regressor: RandomForestRegressor) -> None:
        self._regressor = regressor

    def predict(self, partition: PredictionPartition) -> NDArray[np.float64]:
        """Predict held-out weights; for example, ``predict(partition)`` returns kilograms."""
        return np.asarray(self._regressor.predict(partition.values), dtype=np.float64)


class RandomForestBaseline:
    """Frozen Random Forest baseline; for example, inject it into feature evaluation."""

    name = "random_forest"
    recipe: dict[str, bool | float | int | str | None] = {
        "n_estimators": 500,
        "criterion": "squared_error",
        "bootstrap": True,
        "max_depth": None,
        "min_samples_leaf": 3,
        "min_samples_split": 6,
        "max_features": 0.7,
        "random_state": 44,
    }

    def fit(
        self, partition: TrainingPartition, feature_names: tuple[str, ...]
    ) -> FeaturePredictor:
        """Fit only external-train rows; for example, folds pass one training partition."""
        regressor = RandomForestRegressor(**self.recipe)
        regressor.fit(partition.values, partition.targets_kg)
        return SklearnFeaturePredictor(regressor)


DENSE_BASELINE_RECIPE = DenseTrainingRecipe()


@dataclass(frozen=True)
class DenseTrainingAudit:
    selection_ids: tuple[str, ...]
    stopping_ids: tuple[str, ...]
    retrain_ids: tuple[str, ...]
    selected_epochs: int


@dataclass(frozen=True)
class _FeatureScale:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return (values - self.mean) / self.scale


class DenseFeaturePredictor:
    """Restore kilograms after CUDA inference; for example, use on an outer fold."""

    def __init__(
        self, adapter: DenseFeatureAdapter, model: DenseFeatureNetwork,
        feature_scale: _FeatureScale, target_mean: float, target_scale: float,
    ) -> None:
        self._adapter, self._model, self._feature_scale = adapter, model, feature_scale
        self._target_mean, self._target_scale = target_mean, target_scale

    def predict(self, partition: PredictionPartition) -> NDArray[np.float64]:
        """Predict kilograms; for example, ``predict(outer_fold)`` returns one value per row."""
        values = self._feature_scale.transform(partition.values)
        standardized = self._adapter.predict_array(self._model, values)
        return standardized * self._target_scale + self._target_mean


class DenseFeatureBaseline:
    """Frozen dense baseline with isolated epoch selection and full outer retraining."""

    name = "dense"

    def __init__(
        self, recipe: DenseTrainingRecipe = DENSE_BASELINE_RECIPE,
        adapter: DenseFeatureAdapter | None = None,
    ) -> None:
        self.recipe = recipe
        self._adapter = adapter or DenseFeatureAdapter()
        self.training_audits: list[DenseTrainingAudit] = []

    def fit(
        self, partition: TrainingPartition, feature_names: tuple[str, ...]
    ) -> FeaturePredictor:
        """Fit without outer-fold access; for example, evaluation passes only train rows."""
        selection, stopping = _inner_indices(partition, self.recipe.inner_seed)
        inner_features = _fit_feature_scale(partition.values[selection])
        inner_mean, inner_scale = _fit_target_scale(partition.targets_kg[selection])
        epochs = self._select_epochs(partition, selection, stopping, inner_features,
                                     inner_mean, inner_scale)
        predictor = self._retrain(partition, epochs)
        self.training_audits.append(_training_audit(partition, selection, stopping, epochs))
        return predictor

    def _select_epochs(
        self, partition: TrainingPartition, selection: NDArray[np.int64],
        stopping: NDArray[np.int64], feature_scale: _FeatureScale,
        target_mean: float, target_scale: float,
    ) -> int:
        train_x = feature_scale.transform(partition.values[selection])
        train_y = (partition.targets_kg[selection] - target_mean) / target_scale
        validation_x = feature_scale.transform(partition.values[stopping])
        return self._adapter.select_epoch_count(train_x, train_y, validation_x,
                                                partition.targets_kg[stopping], target_mean,
                                                target_scale, self.recipe)

    def _retrain(self, partition: TrainingPartition, epochs: int) -> DenseFeaturePredictor:
        feature_scale = _fit_feature_scale(partition.values)
        target_mean, target_scale = _fit_target_scale(partition.targets_kg)
        values = feature_scale.transform(partition.values)
        targets = (partition.targets_kg - target_mean) / target_scale
        model = self._adapter.fit_epochs(values, targets, epochs, self.recipe)
        return DenseFeaturePredictor(self._adapter, model, feature_scale, target_mean, target_scale)


def _inner_indices(
    partition: TrainingPartition, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    selection, stopping = next(splitter.split(partition.values, partition.strata))
    return np.asarray(selection, dtype=np.int64), np.asarray(stopping, dtype=np.int64)


def _fit_feature_scale(values: NDArray[np.float64]) -> _FeatureScale:
    mean = np.mean(values, axis=0)
    standard_deviation = np.std(values, axis=0)
    scale = np.where(standard_deviation == 0.0, 1.0, standard_deviation)
    return _FeatureScale(mean, scale)


def _fit_target_scale(targets: NDArray[np.float64]) -> tuple[float, float]:
    mean, standard_deviation = float(np.mean(targets)), float(np.std(targets))
    return mean, standard_deviation if standard_deviation != 0.0 else 1.0


def _training_audit(
    partition: TrainingPartition, selection: NDArray[np.int64],
    stopping: NDArray[np.int64], epochs: int,
) -> DenseTrainingAudit:
    selected_ids = tuple(partition.sample_ids[index] for index in selection)
    stopping_ids = tuple(partition.sample_ids[index] for index in stopping)
    return DenseTrainingAudit(selected_ids, stopping_ids, partition.sample_ids, epochs)
