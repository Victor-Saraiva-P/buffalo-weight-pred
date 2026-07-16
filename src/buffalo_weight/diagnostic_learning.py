from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

from buffalo_weight.canonical_mask import load_canonical_masks
from buffalo_weight.cnn_mask import load_masks
from buffalo_weight.models import ModelParam
from buffalo_weight.split import assign_folds, assign_weight_categories
from buffalo_weight.train import format_metric


LEARNING_FIELDS = [
    "split_seed", "fold", "fraction", "n_train", "train_mae", "validation_mae", "heavy_validation_mae"
]


def _matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    return np.asarray([[float(row[column].replace(",", ".")) for column in columns] for row in rows])


def _subsample_indexes(rows: list[dict[str, str]], fraction: float, seed: int) -> np.ndarray:
    indexes = np.arange(len(rows))
    if fraction >= 1.0:
        return indexes
    labels = [row["weight_category"] for row in rows]
    selected, _ = train_test_split(indexes, train_size=fraction, random_state=seed, stratify=labels)
    return np.asarray(selected)


class LearningCurveRepresentation:
    """Cache deterministic mask representations for repeated learning curves.

    Example: ``LearningCurveRepresentation(rows, columns, masks_dir, 96)``.
    """

    def __init__(self, rows: list[dict[str, str]], columns: list[str], masks_dir: Path, image_size: int) -> None:
        self.rows = rows
        self.geometry = _matrix(rows, columns)
        self.raw_pixels = load_masks(masks_dir, rows, image_size, "stretch").reshape(len(rows), -1)
        canonical = load_canonical_masks(masks_dir, rows, image_size, "letterbox")
        self.canonical_pixels = canonical.reshape(len(rows), -1)
        self.weights = np.asarray([float(row["weight"].replace(",", ".")) for row in rows])
        self.index_by_name = {row["file_name"]: index for index, row in enumerate(rows)}

    def indexes(self, rows: list[dict[str, str]]) -> np.ndarray:
        return np.asarray([self.index_by_name[row["file_name"]] for row in rows])


def _fit_components(pixels: np.ndarray, train: np.ndarray, validation: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    component_count = min(count, len(train) - 1, pixels.shape[1])
    pca = PCA(component_count, svd_solver="randomized", random_state=42)
    return pca.fit_transform(pixels[train]), pca.transform(pixels[validation])


def _fit_learning_model(
    representation: LearningCurveRepresentation, train: np.ndarray, validation: np.ndarray, min_samples_leaf: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    raw_train, raw_validation = _fit_components(representation.raw_pixels, train, validation, 24)
    canonical_train, canonical_validation = _fit_components(
        representation.canonical_pixels, train, validation, 16
    )
    x_train = np.column_stack((representation.geometry[train], raw_train, canonical_train))
    x_validation = np.column_stack((representation.geometry[validation], raw_validation, canonical_validation))
    model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=min_samples_leaf, max_features=1.0, random_state=42)
    model.fit(x_train, np.log(representation.weights[train]))
    return np.exp(model.predict(x_train)), np.exp(model.predict(x_validation))


def learning_curve_rows(
    rows: list[dict[str, str]], representation: LearningCurveRepresentation, split_seeds: list[int]
) -> list[dict[str, str]]:
    """Evaluate train-size effects; for example, ``learning_curve_rows(rows, representation, [0, 1])``."""
    results = []
    for split_seed in split_seeds:
        split_rows = [row.copy() for row in rows]
        assign_weight_categories(split_rows, 10)
        assign_folds(split_rows, 5, split_seed)
        for fold in range(1, 6):
            results.extend(_fold_learning_rows(split_rows, representation, split_seed, fold))
    return results


def _fold_learning_rows(
    split_rows: list[dict[str, str]], representation: LearningCurveRepresentation, split_seed: int, fold: int
) -> list[dict[str, str]]:
    training_rows = [row for row in split_rows if int(row["fold"]) != fold]
    validation_rows = [row for row in split_rows if int(row["fold"]) == fold]
    validation = representation.indexes(validation_rows)
    results = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        local = _subsample_indexes(training_rows, fraction, split_seed * 101 + fold)
        train = representation.indexes([training_rows[index] for index in local])
        train_prediction, validation_prediction = _fit_learning_model(representation, train, validation)
        results.append(_learning_row(split_seed, fold, fraction, train, validation, representation, train_prediction, validation_prediction))
    return results


def _learning_row(
    seed: int, fold: int, fraction: float, train: np.ndarray, validation: np.ndarray,
    representation: LearningCurveRepresentation, train_prediction: np.ndarray, validation_prediction: np.ndarray,
) -> dict[str, str]:
    train_errors = np.abs(train_prediction - representation.weights[train])
    validation_errors = np.abs(validation_prediction - representation.weights[validation])
    heavy = representation.weights[validation] >= np.quantile(representation.weights, 0.8)
    return {
        "split_seed": str(seed), "fold": str(fold), "fraction": str(fraction), "n_train": str(len(train)),
        "train_mae": format_metric(float(train_errors.mean())),
        "validation_mae": format_metric(float(validation_errors.mean())),
        "heavy_validation_mae": format_metric(float(validation_errors[heavy].mean())),
    }


def capacity_gap_rows(
    rows: list[dict[str, str]], representation: LearningCurveRepresentation, split_seeds: list[int]
) -> list[dict[str, str]]:
    """Contrast interpolation and validation; for example, ``capacity_gap_rows(rows, representation, [0])``."""
    results = []
    for split_seed in split_seeds:
        split_rows = [row.copy() for row in rows]
        assign_weight_categories(split_rows, 10)
        assign_folds(split_rows, 5, split_seed)
        for fold in range(1, 6):
            train = representation.indexes([row for row in split_rows if int(row["fold"]) != fold])
            validation = representation.indexes([row for row in split_rows if int(row["fold"]) == fold])
            for leaf in (1, 2):
                train_prediction, validation_prediction = _fit_learning_model(
                    representation, train, validation, leaf
                )
                results.append({
                    "split_seed": str(split_seed), "fold": str(fold), "min_samples_leaf": str(leaf),
                    "train_mae": format_metric(float(np.mean(np.abs(train_prediction - representation.weights[train])))),
                    "validation_mae": format_metric(float(np.mean(np.abs(validation_prediction - representation.weights[validation])))),
                })
    return results
