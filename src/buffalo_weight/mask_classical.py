from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from buffalo_weight.cnn_mask import load_masks
from buffalo_weight.models import ModelParam
from buffalo_weight.split import parse_weight


MASK_REPRESENTATIONS = frozenset({"pixels_pca", "shape_profile"})
MASK_ESTIMATORS = frozenset({"extra_trees", "kernel_ridge", "ridge", "svr"})


def shape_profile_features(masks: np.ndarray) -> np.ndarray:
    """Describe silhouette occupancy and edges; for example, ``shape_profile_features(masks)``."""
    size = masks.shape[1]
    scale = float(max(size - 1, 1))
    row_present = masks.any(axis=2)
    column_present = masks.any(axis=1)
    row_first = np.where(row_present, masks.argmax(axis=2) / scale, 0.0)
    row_last = np.where(row_present, (size - 1 - masks[:, :, ::-1].argmax(axis=2)) / scale, 0.0)
    column_first = np.where(column_present, masks.argmax(axis=1) / scale, 0.0)
    column_last = np.where(column_present, (size - 1 - masks[:, ::-1, :].argmax(axis=1)) / scale, 0.0)
    return np.column_stack(
        (masks.mean(axis=2), row_first, row_last, masks.mean(axis=1), column_first, column_last)
    )


def _regressor(params: dict[str, ModelParam]) -> object:
    estimator = str(params["estimator"])
    if estimator == "ridge":
        return Ridge(alpha=float(params.get("alpha", 1.0)))
    if estimator == "kernel_ridge":
        return KernelRidge(alpha=float(params.get("alpha", 1.0)), kernel="rbf", gamma=params.get("gamma"))
    if estimator == "svr":
        return SVR(
            C=float(params.get("c", 10.0)),
            epsilon=float(params.get("epsilon", 0.1)),
            gamma=params.get("gamma", "scale"),
        )
    if estimator == "extra_trees":
        names = ("n_estimators", "random_state", "min_samples_leaf", "max_features")
        return ExtraTreesRegressor(**{name: params[name] for name in names if name in params})
    raise ValueError(f"mask estimator was {estimator!r}; expected one of {sorted(MASK_ESTIMATORS)}")


def regression_pipeline(params: dict[str, ModelParam], input_width: int) -> Pipeline:
    """Build a fold-fitted mask regressor; for example, ``regression_pipeline(params, 128)``."""
    steps = []
    component_count = min(int(params.get("n_components", 0)), input_width)
    if component_count:
        steps.append(("pca", PCA(component_count, svd_solver="randomized", random_state=int(params["random_state"]))))
    steps.extend((('scale', StandardScaler()), ('regressor', _regressor(params))))
    return Pipeline(steps)


class MaskFeatureRegressor:
    """Regress weight from deterministic binary-mask descriptors.

    Example: ``MaskFeatureRegressor(Path("masks"), params).fit(rows)``.
    """

    def __init__(self, masks_dir: Path, params: dict[str, ModelParam]) -> None:
        self.masks_dir = masks_dir
        self.params = params
        self.image_size = int(params["image_size"])
        self.resize_mode = str(params.get("resize_mode", "letterbox"))
        self.representation = str(params["representation"])
        self.model: Pipeline | None = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _features(self, rows: list[dict[str, str]]) -> np.ndarray:
        masks = load_masks(self.masks_dir, rows, self.image_size, self.resize_mode)
        if self.representation == "shape_profile":
            return shape_profile_features(masks)
        if self.representation == "pixels_pca":
            return masks.reshape(len(rows), -1)
        raise ValueError(
            f"mask representation was {self.representation!r}; expected one of {sorted(MASK_REPRESENTATIONS)}"
        )

    def fit(self, rows: list[dict[str, str]]) -> None:
        features = self._features(rows)
        weights = np.asarray([parse_weight(row["weight"], row["file_name"]) for row in rows])
        self.y_mean = float(weights.mean())
        self.y_std = float(weights.std() or 1.0)
        params = {**self.params, "n_components": min(int(self.params.get("n_components", 0)), len(rows) - 1)}
        self.model = regression_pipeline(params, features.shape[1])
        self.model.fit(features, (weights - self.y_mean) / self.y_std)

    def predict(self, rows: list[dict[str, str]]) -> np.ndarray:
        """Predict one weight per mask; for example, ``model.predict(rows)``."""
        if self.model is None:
            raise ValueError("mask_feature model has not been fitted")
        scaled = self.model.predict(self._features(rows))
        return np.asarray(scaled * self.y_std + self.y_mean, dtype=float)
