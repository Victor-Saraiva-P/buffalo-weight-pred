from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, rotate, shift
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

from buffalo_weight.canonical_mask import canonicalize_mask
from buffalo_weight.cnn_mask import load_masks
from buffalo_weight.diagnostic_metrics import metric_summary
from buffalo_weight.feature_calculators import calculate_mask_features


PERTURBATIONS = ("original", "erosion_1", "erosion_2", "dilation_1", "dilation_2", "shift_4", "rotation_5")


def perturb_masks(masks: np.ndarray, scenario: str) -> np.ndarray:
    """Apply synthetic segmentation noise; for example, ``perturb_masks(masks, "erosion_1")``."""
    if scenario == "original":
        return masks.copy()
    if scenario.startswith("erosion_"):
        iterations = int(scenario.removeprefix("erosion_"))
        return np.stack([binary_erosion(mask, iterations=iterations) for mask in masks]).astype(np.float32)
    if scenario.startswith("dilation_"):
        iterations = int(scenario.removeprefix("dilation_"))
        return np.stack([binary_dilation(mask, iterations=iterations) for mask in masks]).astype(np.float32)
    if scenario == "shift_4":
        return np.stack([shift(mask, (0, 4), order=0, mode="constant") for mask in masks]).astype(np.float32)
    if scenario == "rotation_5":
        return np.stack([rotate(mask, 5, reshape=False, order=0) for mask in masks]).astype(np.float32)
    raise ValueError(f"perturbation scenario was {scenario!r}; expected one of {PERTURBATIONS}")


def _geometry(masks: np.ndarray, columns: list[str]) -> np.ndarray:
    feature_rows = [calculate_mask_features(mask > 0) for mask in masks]
    return np.asarray([[row[column] for column in columns] for row in feature_rows])


def _canonical_pixels(masks: np.ndarray, image_size: int) -> np.ndarray:
    canonical = [canonicalize_mask(mask > 0, image_size, "letterbox") for mask in masks]
    return np.asarray(canonical).reshape(len(masks), -1)


class RobustnessFoldModel:
    def __init__(self, masks: np.ndarray, weights: np.ndarray, train: np.ndarray, columns: list[str]) -> None:
        self.image_size = masks.shape[1]
        self.columns = columns
        self.raw_pca = PCA(24, svd_solver="randomized", random_state=42)
        self.canonical_pca = PCA(16, svd_solver="randomized", random_state=42)
        raw = self.raw_pca.fit_transform(masks[train].reshape(len(train), -1))
        canonical = self.canonical_pca.fit_transform(_canonical_pixels(masks[train], self.image_size))
        features = np.column_stack((_geometry(masks[train], columns), raw, canonical))
        self.model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
        self.model.fit(features, np.log(weights[train]))

    def predict(self, masks: np.ndarray) -> np.ndarray:
        """Predict perturbed masks; for example, ``model.predict(masks)``."""
        raw = self.raw_pca.transform(masks.reshape(len(masks), -1))
        canonical = self.canonical_pca.transform(_canonical_pixels(masks, self.image_size))
        features = np.column_stack((_geometry(masks, self.columns), raw, canonical))
        return np.exp(self.model.predict(features))


def segmentation_robustness(
    masks_dir: Path, rows: list[dict[str, str]], columns: list[str]
) -> list[dict[str, str]]:
    """Measure OOF sensitivity to synthetic mask noise; for example, ``segmentation_robustness(path, rows, columns)``."""
    masks = load_masks(masks_dir, rows, 96, "stretch")
    weights = np.asarray([float(row["weight"]) for row in rows])
    folds = np.asarray([int(row["fold"]) for row in rows])
    predictions = {scenario: np.zeros(len(rows)) for scenario in PERTURBATIONS}
    for fold in sorted(set(folds)):
        train, validation = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model = RobustnessFoldModel(masks, weights, train, columns)
        for scenario in PERTURBATIONS:
            predictions[scenario][validation] = model.predict(perturb_masks(masks[validation], scenario))
    original = predictions["original"]
    return [
        {
            "scenario": scenario,
            **metric_summary(weights, predicted),
            "mean_prediction_change": str(float(np.mean(np.abs(predicted - original)))),
            "p90_prediction_change": str(float(np.quantile(np.abs(predicted - original), 0.9))),
        }
        for scenario, predicted in predictions.items()
    ]
