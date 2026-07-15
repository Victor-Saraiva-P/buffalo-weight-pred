from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from buffalo_weight.cnn_mask import load_masks
from buffalo_weight.models import ModelParam
from buffalo_weight.split import parse_weight


class PcaSvrMaskRegressor:
    def __init__(self, masks_dir: Path, params: dict[str, ModelParam]) -> None:
        self.masks_dir = masks_dir
        self.image_size = int(params["image_size"])
        self.resize_mode = str(params.get("resize_mode", "letterbox"))
        self.n_components = int(params["n_components"])
        self.random_state = int(params["random_state"])
        self.cost = float(params.get("c", 10.0))
        self.epsilon = float(params.get("epsilon", 0.1))
        gamma = params.get("gamma", "scale")
        self.gamma = str(gamma) if isinstance(gamma, str) else float(gamma)
        self.model: Pipeline | None = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _build_model(self, component_count: int) -> Pipeline:
        return Pipeline(
            [
                ("pca", PCA(component_count, svd_solver="randomized", random_state=self.random_state)),
                ("scale", StandardScaler()),
                ("svr", SVR(C=self.cost, epsilon=self.epsilon, gamma=self.gamma)),
            ]
        )

    def fit(self, rows: list[dict[str, str]]) -> None:
        masks = load_masks(self.masks_dir, rows, self.image_size, self.resize_mode)
        pixels = masks.reshape(len(rows), -1)
        weights = np.asarray(
            [parse_weight(row["weight"], row["file_name"]) for row in rows], dtype=np.float32
        )
        self.y_mean = float(weights.mean())
        self.y_std = float(weights.std() or 1.0)
        component_count = min(self.n_components, len(rows) - 1, pixels.shape[1])
        if component_count < 1:
            raise ValueError(f"pca_svr_mask received {len(rows)} rows; expected at least 2 rows")
        self.model = self._build_model(component_count)
        self.model.fit(pixels, (weights - self.y_mean) / self.y_std)

    def predict(self, rows: list[dict[str, str]]) -> np.ndarray:
        if self.model is None:
            raise ValueError("pca_svr_mask model has not been fitted")
        masks = load_masks(self.masks_dir, rows, self.image_size, self.resize_mode)
        pixels = masks.reshape(len(rows), -1)
        predictions = self.model.predict(pixels)
        return np.asarray(predictions * self.y_std + self.y_mean, dtype=float)
