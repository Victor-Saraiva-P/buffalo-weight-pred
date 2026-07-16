from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

from buffalo_weight.canonical_mask import load_canonical_masks
from buffalo_weight.cnn_mask import load_masks
from buffalo_weight.models import ModelParam
from buffalo_weight.split import parse_weight
from buffalo_weight.target_transform import (
    inverse_target,
    inverse_target_power,
    transform_target,
    transform_target_power,
)


def _feature_value(row: dict[str, str], column: str) -> float:
    raw_value = row.get(column, "")
    try:
        value = float(raw_value.replace(",", "."))
    except ValueError as error:
        raise ValueError(
            f"feature {column} for {row.get('file_name', '')} was {raw_value!r}; expected a finite number"
        ) from error
    if math.isfinite(value):
        return value
    raise ValueError(
        f"feature {column} for {row.get('file_name', '')} was {raw_value!r}; expected a finite number"
    )


class PcaFeatureFusionRegressor:
    """Fuse geometric features with fold-fitted PCA mask components.

    Example: ``PcaFeatureFusionRegressor(Path("masks"), params).fit(rows, ["area"])``.
    """

    def __init__(self, masks_dir: Path, params: dict[str, ModelParam]) -> None:
        self.masks_dir = masks_dir
        self.image_size = int(params["image_size"])
        self.resize_mode = str(params.get("resize_mode", "letterbox"))
        self.n_components = int(params["n_components"])
        self.random_state = int(params["random_state"])
        self.target_transform = str(params.get("target_transform", "identity"))
        self.target_power = float(params["target_power"]) if "target_power" in params else None
        self.canonical_components = int(params.get("canonical_components", 0))
        self.canonical_resize_mode = str(params.get("canonical_resize_mode", "letterbox"))
        self.heavy_sample_weight = float(params.get("heavy_sample_weight", 1.0))
        self.heavy_quantile = float(params.get("heavy_quantile", 0.8))
        self.model = ExtraTreesRegressor(**self._forest_params(params))
        self.pca: PCA | None = None
        self.canonical_pca: PCA | None = None

    def _forest_params(self, params: dict[str, ModelParam]) -> dict[str, ModelParam]:
        names = ("n_estimators", "random_state", "max_depth", "min_samples_leaf", "max_features")
        return {name: params[name] for name in names if name in params}

    def _pixels(self, rows: list[dict[str, str]]) -> np.ndarray:
        masks = load_masks(self.masks_dir, rows, self.image_size, self.resize_mode)
        return masks.reshape(len(rows), -1)

    def _geometry(self, rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
        return np.asarray([[_feature_value(row, column) for column in columns] for row in rows])

    def _canonical_pixels(self, rows: list[dict[str, str]]) -> np.ndarray:
        masks = load_canonical_masks(
            self.masks_dir, rows, self.image_size, self.canonical_resize_mode
        )
        return masks.reshape(len(rows), -1)

    def _component_features(self, rows: list[dict[str, str]]) -> list[np.ndarray]:
        if self.pca is None:
            raise ValueError("pca_feature_fusion model has not been fitted")
        components = [self.pca.transform(self._pixels(rows))]
        if self.canonical_pca is not None:
            components.append(self.canonical_pca.transform(self._canonical_pixels(rows)))
        return components

    def _fused_features(self, rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
        return np.column_stack((self._geometry(rows, columns), *self._component_features(rows)))

    def _fit_canonical_pca(self, rows: list[dict[str, str]]) -> np.ndarray | None:
        component_count = min(self.canonical_components, len(rows) - 1, self.image_size**2)
        if component_count < 1:
            return None
        self.canonical_pca = PCA(component_count, svd_solver="randomized", random_state=self.random_state)
        return self.canonical_pca.fit_transform(self._canonical_pixels(rows))

    def _transform_weights(self, weights: np.ndarray) -> np.ndarray:
        if self.target_power is not None:
            return transform_target_power(weights, self.target_power)
        return transform_target(weights, self.target_transform)

    def _inverse_weights(self, weights: np.ndarray) -> np.ndarray:
        if self.target_power is not None:
            return inverse_target_power(weights, self.target_power)
        return inverse_target(weights, self.target_transform)

    def _sample_weights(self, weights: np.ndarray) -> np.ndarray:
        if not 0.0 < self.heavy_quantile < 1.0 or self.heavy_sample_weight < 1.0:
            raise ValueError(
                f"heavy weighting was quantile={self.heavy_quantile}, weight={self.heavy_sample_weight}; "
                "expected 0 < heavy_quantile < 1 and heavy_sample_weight >= 1"
            )
        threshold = float(np.quantile(weights, self.heavy_quantile))
        return np.where(weights >= threshold, self.heavy_sample_weight, 1.0)

    def fit(self, rows: list[dict[str, str]], columns: list[str]) -> None:
        component_count = min(self.n_components, len(rows) - 1, self.image_size**2)
        if component_count < 1:
            raise ValueError(f"pca_feature_fusion received {len(rows)} rows; expected at least 2 rows")
        self.pca = PCA(component_count, svd_solver="randomized", random_state=self.random_state)
        components = self.pca.fit_transform(self._pixels(rows))
        canonical = self._fit_canonical_pca(rows)
        feature_parts = (self._geometry(rows, columns), components, canonical)
        features = np.column_stack(tuple(part for part in feature_parts if part is not None))
        weights = np.asarray([parse_weight(row["weight"], row["file_name"]) for row in rows])
        self.model.fit(features, self._transform_weights(weights), sample_weight=self._sample_weights(weights))

    def predict(self, rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
        """Predict one weight per row; for example, ``model.predict(rows, ["area"])``."""
        transformed = np.asarray(self.model.predict(self._fused_features(rows, columns)), dtype=float)
        return self._inverse_weights(transformed)
