from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from buffalo_weight.cnn_mask import EarlyStopping, load_mask
from buffalo_weight.models import parse_model_configs


class ModelConfigTest(unittest.TestCase):
    def test_parses_cnn_mask_model_config(self) -> None:
        configs = parse_model_configs(
            {
                "model_configs": {
                    "cnn_mask_baseline": {
                        "model": "cnn_mask",
                        "params": {
                            "epochs": 5,
                            "batch_size": 4,
                            "learning_rate": 0.001,
                            "image_size": 64,
                            "random_state": 42,
                        },
                    }
                }
            }
        )

        self.assertEqual(configs[0].name, "cnn_mask_baseline")
        self.assertEqual(configs[0].model, "cnn_mask")

    def test_rejects_unknown_model_param(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported params"):
            parse_model_configs(
                {
                    "model_configs": {
                        "cnn_mask_baseline": {
                            "model": "cnn_mask",
                            "params": {
                                "epochs": 5,
                                "batch_size": 4,
                                "learning_rate": 0.001,
                                "image_size": 64,
                                "random_state": 42,
                                "bad_param": 1,
                            },
                        }
                    }
                }
            )

    def test_rejects_non_scalar_model_param(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected a scalar"):
            parse_model_configs(
                {
                    "model_configs": {
                        "random_forest_baseline": {
                            "model": "random_forest",
                            "params": {"n_estimators": [100], "random_state": 42},
                        }
                    }
                }
            )


class CnnMaskTest(unittest.TestCase):
    def test_load_mask_accepts_binary_black_white_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.asarray([[0, 255], [255, 0]], dtype=np.uint8)).save(path)

            mask = load_mask(path, image_size=2)

        np.testing.assert_array_equal(mask, np.asarray([[0, 1], [1, 0]], dtype=np.float32))

    def test_load_mask_rejects_non_binary_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.asarray([[0, 128], [255, 0]], dtype=np.uint8)).save(path)

            with self.assertRaisesRegex(ValueError, "mask must be binary black/white"):
                load_mask(path, image_size=2)

    def test_early_stopping_restores_parameters_from_lowest_loss(self) -> None:
        import torch
        from torch import nn

        model = nn.Linear(1, 1, bias=False)
        early_stopping = EarlyStopping(patience=2)
        with torch.no_grad():
            model.weight.fill_(1.0)
        self.assertFalse(early_stopping.observe(model, loss=3.0))
        with torch.no_grad():
            model.weight.fill_(2.0)
        self.assertFalse(early_stopping.observe(model, loss=2.0))
        with torch.no_grad():
            model.weight.fill_(9.0)
        self.assertFalse(early_stopping.observe(model, loss=4.0))
        self.assertTrue(early_stopping.observe(model, loss=5.0))

        early_stopping.restore(model)

        self.assertEqual(float(model.weight.item()), 2.0)


if __name__ == "__main__":
    unittest.main()
