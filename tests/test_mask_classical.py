from __future__ import annotations

import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np

from buffalo_weight.mask_classical import MaskFeatureRegressor, shape_profile_features
from buffalo_weight.pretrained_mask_embedding import PretrainedMaskEmbeddingRegressor


class FakeEmbeddingDeviceResolver:
    def __init__(self) -> None:
        """Start without a requested official neural device."""

        self.requested_device: str | None = None

    def __call__(self, requested_device: str, cuda_available: Callable[[], bool]) -> str:
        """Record the official request while executing tensors on fake CPU compute."""

        self.requested_device = requested_device
        return "cpu"


class FakeMaskLoader:
    def __call__(
        self, masks_dir: Path, rows: list[dict[str, str]], image_size: int, resize_mode: str
    ) -> np.ndarray:
        """Return deterministic masks without touching the filesystem."""
        masks = np.zeros((len(rows), image_size, image_size), dtype=np.float32)
        for index, mask in enumerate(masks):
            mask[1:-1, 1 : index + 2] = 1.0
        return masks


class FakeEmbeddingNetworkBuilder:
    def __call__(self, architecture: str) -> object:
        """Return a tiny frozen-compatible embedding network."""
        import torch

        return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())


class MaskClassicalTest(unittest.TestCase):
    def test_shape_profiles_encode_six_signatures_per_axis_position(self) -> None:
        masks = np.zeros((2, 4, 4), dtype=np.float32)
        masks[:, 1:3, 1:3] = 1

        profiles = shape_profile_features(masks)

        self.assertEqual(profiles.shape, (2, 24))
        self.assertTrue(np.isfinite(profiles).all())

    def test_mask_profile_regressor_fits_binary_masks(self) -> None:
        rows = mask_regression_rows()
        params = {
            "image_size": 8,
            "representation": "shape_profile",
            "estimator": "ridge",
            "alpha": 1.0,
            "random_state": 42,
        }
        model = MaskFeatureRegressor(Path("unused"), params, FakeMaskLoader())

        model.fit(rows)
        predictions = model.predict(rows)

        self.assertEqual(predictions.shape, (8,))

    def test_pretrained_embedding_regressor_uses_frozen_mask_embeddings(self) -> None:
        rows = mask_regression_rows()
        params = {
            "image_size": 8, "architecture": "resnet18", "estimator": "ridge",
            "n_components": 2, "random_state": 42,
        }
        device_resolver = FakeEmbeddingDeviceResolver()
        model = PretrainedMaskEmbeddingRegressor(
            Path("unused"), params, "cuda", device_resolver,
            FakeEmbeddingNetworkBuilder(), FakeMaskLoader(),
        )

        model.fit(rows)
        predictions = model.predict(rows)

        self.assertEqual(predictions.shape, (8,))
        self.assertEqual(device_resolver.requested_device, "cuda")


def mask_regression_rows() -> list[dict[str, str]]:
    """Create labelled mask rows; for example, ``mask_regression_rows()`` returns eight."""
    rows = []
    for index in range(8):
        rows.append({"file_name": f"profile-{index}", "weight": str(100 + index * 10)})
    return rows


if __name__ == "__main__":
    unittest.main()
