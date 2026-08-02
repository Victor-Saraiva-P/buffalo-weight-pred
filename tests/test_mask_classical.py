from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

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


class MaskClassicalTest(unittest.TestCase):
    def test_shape_profiles_encode_six_signatures_per_axis_position(self) -> None:
        masks = np.zeros((2, 4, 4), dtype=np.float32)
        masks[:, 1:3, 1:3] = 1

        profiles = shape_profile_features(masks)

        self.assertEqual(profiles.shape, (2, 24))
        self.assertTrue(np.isfinite(profiles).all())

    def test_mask_profile_regressor_fits_binary_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir, rows = mask_regression_fixture(Path(directory))
            params = {
                "image_size": 8,
                "representation": "shape_profile",
                "estimator": "ridge",
                "alpha": 1.0,
                "random_state": 42,
            }
            model = MaskFeatureRegressor(masks_dir, params)

            model.fit(rows)
            predictions = model.predict(rows)

        self.assertEqual(predictions.shape, (8,))

    def test_pretrained_embedding_regressor_uses_frozen_mask_embeddings(self) -> None:
        import torch
        with tempfile.TemporaryDirectory() as directory:
            masks_dir, rows = mask_regression_fixture(Path(directory))
            params = {
                "image_size": 8, "architecture": "resnet18", "estimator": "ridge",
                "n_components": 2, "random_state": 42,
            }
            network = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
            device_resolver = FakeEmbeddingDeviceResolver()
            model = PretrainedMaskEmbeddingRegressor(
                masks_dir, params, "cuda", device_resolver=device_resolver
            )

            with patch("buffalo_weight.pretrained_mask_embedding.build_embedding_network", return_value=network):
                model.fit(rows)
                predictions = model.predict(rows)

        self.assertEqual(predictions.shape, (8,))
        self.assertEqual(device_resolver.requested_device, "cuda")


def mask_regression_fixture(masks_dir: Path) -> tuple[Path, list[dict[str, str]]]:
    """Create varied binary masks; for example, ``mask_regression_fixture(Path(tmp))``."""
    rows = []
    for index in range(8):
        pixels = np.zeros((8, 8), dtype=np.uint8)
        pixels[1:7, 1 : index + 1] = 255
        Image.fromarray(pixels).save(masks_dir / f"profile-{index}.png")
        rows.append({"file_name": f"profile-{index}", "weight": str(100 + index * 10)})
    return masks_dir, rows


if __name__ == "__main__":
    unittest.main()
