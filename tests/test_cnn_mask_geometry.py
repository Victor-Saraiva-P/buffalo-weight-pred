from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from buffalo_weight.cnn_mask_geometry import CnnMaskGeometryRegressor
from buffalo_weight.pure_geometry_evaluation import PURE_GEOMETRY_FEATURES


class TinyFusionNetwork(nn.Module):
    def __init__(self, input_channels: int, feature_count: int) -> None:
        super().__init__()
        self.output = nn.Linear(input_channels + feature_count, 1)

    def forward(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        mask_means = masks.mean(dim=(2, 3))
        return self.output(torch.cat((mask_means, features), dim=1))


class RecordingNetworkBuilder:
    architecture = ""
    pretrained = False
    fine_tune_mode = ""

    def __call__(
        self,
        input_channels: int,
        feature_count: int,
        architecture: str,
        pretrained: bool,
        fine_tune_mode: str,
    ) -> nn.Module:
        type(self).architecture = architecture
        type(self).pretrained = pretrained
        type(self).fine_tune_mode = fine_tune_mode
        return TinyFusionNetwork(input_channels, feature_count)


class CnnMaskGeometryRegressorTest(unittest.TestCase):
    def test_fits_masks_and_pure_geometry_then_predicts_each_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            rows = self._mask_rows(masks_dir)
            regressor = CnnMaskGeometryRegressor(masks_dir, self._params(), "cpu")

            regressor.fit(rows[:6], list(PURE_GEOMETRY_FEATURES), rows[6:7])
            predictions = regressor.predict(rows[7:], list(PURE_GEOMETRY_FEATURES))

        self.assertEqual(predictions.shape, (1,))
        self.assertTrue(np.isfinite(predictions).all())

    def test_passes_pretrained_backbone_recipe_to_injected_builder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            rows = self._mask_rows(masks_dir)
            params = {
                **self._params(),
                "architecture": "resnet18",
                "pretrained": True,
                "fine_tune_mode": "last_block",
            }
            regressor = CnnMaskGeometryRegressor(
                masks_dir, params, "cpu", RecordingNetworkBuilder()
            )

            regressor.fit(rows[:6], list(PURE_GEOMETRY_FEATURES))

        self.assertEqual(RecordingNetworkBuilder.architecture, "resnet18")
        self.assertTrue(RecordingNetworkBuilder.pretrained)
        self.assertEqual(RecordingNetworkBuilder.fine_tune_mode, "last_block")

    def _mask_rows(self, masks_dir: Path) -> list[dict[str, str]]:
        rows = []
        for index in range(8):
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 2 : index + 6] = 255
            Image.fromarray(mask).save(masks_dir / f"mask-{index}.png")
            feature_values = {
                feature: str(index + offset + 1)
                for offset, feature in enumerate(PURE_GEOMETRY_FEATURES)
            }
            rows.append(
                {
                    "file_name": f"mask-{index}",
                    "weight": str(100 + index * 10),
                    **feature_values,
                }
            )
        return rows

    def _params(self) -> dict[str, bool | float | int | str]:
        return {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "image_size": 16,
            "random_state": 42,
            "architecture": "residual",
            "input_representation": "geometry_channels",
        }


if __name__ == "__main__":
    unittest.main()
