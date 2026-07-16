from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from buffalo_weight.cnn_architectures import build_mask_network
from buffalo_weight.cnn_mask import (
    EarlyStopping,
    _translate_mask,
    augment_batch,
    geometry_channels,
    load_mask,
    load_masks,
    resolve_device,
)
from buffalo_weight.models import (
    EXTRA_TREES_MODEL,
    RANDOM_FOREST_MODEL,
    ModelConfig,
    build_model,
    parse_model_configs,
    validate_unique_model_configs,
    xgboost_compute_params,
)
from buffalo_weight.mask_classical import MaskFeatureRegressor, shape_profile_features
from buffalo_weight.pca_feature_fusion import PcaFeatureFusionRegressor
from buffalo_weight.pca_svr_mask import PcaSvrMaskRegressor
from buffalo_weight.pretrained_mask_embedding import PretrainedMaskEmbeddingRegressor


class ModelConfigTest(unittest.TestCase):
    def test_rejects_duplicate_model_configuration_names(self) -> None:
        configs = [
            ModelConfig("same", RANDOM_FOREST_MODEL, {"n_estimators": 2, "random_state": 1}),
            ModelConfig("same", EXTRA_TREES_MODEL, {"n_estimators": 2, "random_state": 1}),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate model configuration names"):
            validate_unique_model_configs(configs)

    def test_classical_model_can_transform_target_inside_fit(self) -> None:
        model = build_model(
            ModelConfig(
                "forest_cube_root",
                "random_forest",
                {"n_estimators": 10, "random_state": 42, "target_transform": "cube_root"},
            )
        )
        features = np.arange(12, dtype=float).reshape(6, 2)
        weights = np.asarray([64.0, 125.0, 216.0, 343.0, 512.0, 729.0])

        model.fit(features, weights)
        predictions = model.predict(features)

        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions > 0).all())

    def test_builds_additional_classical_models(self) -> None:
        configs = [
            ModelConfig("extra", "extra_trees", {"n_estimators": 5, "random_state": 42}),
            ModelConfig("hist", "hist_gradient_boosting", {"random_state": 42}),
        ]

        models = [build_model(config) for config in configs]

        self.assertEqual(
            [type(model).__name__ for model in models],
            ["ExtraTreesRegressor", "HistGradientBoostingRegressor"],
        )

    def test_xgboost_uses_cuda_histogram_when_available(self) -> None:
        self.assertEqual(
            xgboost_compute_params(cuda_available=True, cuda_build=True),
            {"device": "cuda", "tree_method": "hist"},
        )

    def test_xgboost_falls_back_to_cpu_without_cuda(self) -> None:
        self.assertEqual(
            xgboost_compute_params(cuda_available=False, cuda_build=True),
            {"device": "cpu", "tree_method": "hist"},
        )

    def test_xgboost_falls_back_to_cpu_without_cuda_build(self) -> None:
        self.assertEqual(
            xgboost_compute_params(cuda_available=True, cuda_build=False),
            {"device": "cpu", "tree_method": "hist"},
        )

    def test_xgboost_prediction_avoids_mismatched_device_fallback(self) -> None:
        model = build_model(
            ModelConfig("xgboost_test", "xgboost", {"n_estimators": 2, "random_state": 42})
        )
        features = np.arange(20, dtype=float).reshape(10, 2)
        targets = np.arange(10, dtype=float)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            model.fit(features, targets)
            predictions = model.predict(features)

        self.assertEqual(predictions.shape, (10,))
        self.assertFalse(any("mismatched devices" in str(item.message) for item in caught_warnings))

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

    def test_parses_pca_svr_mask_model_config(self) -> None:
        configs = parse_model_configs(
            {
                "model_configs": {
                    "pca_svr_mask_baseline": {
                        "model": "pca_svr_mask",
                        "params": {"image_size": 128, "n_components": 16, "random_state": 42},
                    }
                }
            }
        )

        self.assertEqual(configs[0].name, "pca_svr_mask_baseline")
        self.assertEqual(configs[0].model, "pca_svr_mask")

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
    def test_auto_device_uses_cuda_when_available(self) -> None:
        self.assertEqual(resolve_device("auto", lambda: True), "cuda")

    def test_auto_device_falls_back_to_cpu(self) -> None:
        self.assertEqual(resolve_device("auto", lambda: False), "cpu")

    def test_explicit_cuda_requires_available_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA is not available"):
            resolve_device("cuda", lambda: False)

    def test_mask_network_architectures_predict_one_weight_per_mask(self) -> None:
        import torch

        masks = torch.zeros((2, 1, 64, 64))

        for architecture in ("baseline", "residual"):
            predictions = build_mask_network(architecture)(masks)
            self.assertEqual(tuple(predictions.shape), (2, 1))

    def test_mask_networks_accept_geometry_channels(self) -> None:
        import torch

        inputs = torch.zeros((2, 3, 64, 64))

        for architecture in ("baseline", "residual", "resnet18"):
            network = build_mask_network(architecture, pretrained=False, input_channels=3)
            network.eval()
            self.assertEqual(tuple(network(inputs).shape), (2, 1))

    def test_rejects_unknown_mask_network_architecture(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown.*baseline.*residual"):
            build_mask_network("unknown")

    def test_mobilenet_predicts_from_one_channel_masks(self) -> None:
        import torch

        network = build_mask_network(
            "mobilenet_v3_small", pretrained=False, fine_tune_mode="head"
        )

        predictions = network(torch.zeros((2, 1, 64, 64)))

        self.assertEqual(tuple(predictions.shape), (2, 1))

    def test_pretrained_architectures_predict_from_one_channel_masks(self) -> None:
        import torch

        masks = torch.zeros((2, 1, 64, 64))

        for architecture in ("efficientnet_b0", "resnet18"):
            network = build_mask_network(
                architecture, pretrained=False, fine_tune_mode="last_block"
            )
            network.eval()
            predictions = network(masks)
            self.assertEqual(tuple(predictions.shape), (2, 1))

    def test_pretrained_architectures_only_unfreeze_last_stage(self) -> None:
        for architecture in ("efficientnet_b0", "resnet18"):
            network = build_mask_network(
                architecture, pretrained=False, fine_tune_mode="last_block"
            )
            feature_parameters = [
                parameter.requires_grad
                for name, parameter in network.backbone.named_parameters()
                if "classifier" not in name and not name.startswith("fc.")
            ]
            self.assertTrue(any(feature_parameters))
            self.assertFalse(all(feature_parameters))

    def test_mobilenet_fine_tune_modes_select_backbone_parameters(self) -> None:
        head_only = build_mask_network(
            "mobilenet_v3_small", pretrained=False, fine_tune_mode="head"
        )
        last_block = build_mask_network(
            "mobilenet_v3_small", pretrained=False, fine_tune_mode="last_block"
        )

        self.assertFalse(any(parameter.requires_grad for parameter in head_only.backbone.features.parameters()))
        trainable = [parameter.requires_grad for parameter in last_block.backbone.features.parameters()]
        self.assertTrue(any(trainable))
        self.assertFalse(all(trainable))

    def test_mobilenet_keeps_batch_norm_statistics_frozen(self) -> None:
        import torch

        network = build_mask_network(
            "mobilenet_v3_small", pretrained=False, fine_tune_mode="last_block"
        )

        network.train()

        batch_norms = [module for module in network.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        self.assertTrue(batch_norms)
        self.assertTrue(all(not module.training for module in batch_norms))

    def test_rejects_unknown_mobilenet_fine_tune_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown.*head.*last_block"):
            build_mask_network(
                "mobilenet_v3_small", pretrained=False, fine_tune_mode="unknown"
            )

    def test_load_mask_accepts_binary_black_white_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.asarray([[0, 255], [255, 0]], dtype=np.uint8)).save(path)

            mask = load_mask(path, image_size=2)

        np.testing.assert_array_equal(mask, np.asarray([[0, 1], [1, 0]], dtype=np.float32))

    def test_load_mask_letterboxes_without_distorting_aspect_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.full((2, 4), 255, dtype=np.uint8)).save(path)

            mask = load_mask(path, image_size=4)

        expected = np.asarray(
            [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.float32
        )
        np.testing.assert_array_equal(mask, expected)

    def test_load_mask_can_stretch_to_square(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.full((2, 4), 255, dtype=np.uint8)).save(path)

            mask = load_mask(path, image_size=4, resize_mode="stretch")

        np.testing.assert_array_equal(mask, np.ones((4, 4), dtype=np.float32))

    def test_load_mask_rejects_unknown_resize_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path)

            with self.assertRaisesRegex(ValueError, "unknown.*letterbox.*stretch"):
                load_mask(path, image_size=4, resize_mode="unknown")

    def test_load_masks_adds_one_array_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(masks_dir / "black.png")
            Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(masks_dir / "white.png")

            masks = load_masks(masks_dir, [{"file_name": "black"}, {"file_name": "white"}], image_size=2)

        self.assertEqual(masks.shape, (2, 2, 2))
        self.assertEqual(float(masks[0].sum()), 0.0)
        self.assertEqual(float(masks[1].sum()), 4.0)

    def test_geometry_channels_include_binary_edge_and_normalized_distance(self) -> None:
        masks = np.zeros((1, 5, 5), dtype=np.float32)
        masks[:, 1:4, 1:4] = 1

        channels = geometry_channels(masks)

        self.assertEqual(channels.shape, (1, 3, 5, 5))
        np.testing.assert_array_equal(channels[0, 0], masks[0])
        self.assertEqual(float(channels[0, 1].sum()), 8.0)
        self.assertEqual(float(channels[0, 2].max()), 1.0)

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

    def test_augmentation_samples_transform_per_mask(self) -> None:
        import torch

        batch = torch.zeros((8, 1, 16, 16))
        batch[:, :, 4:8, 5:9] = 1

        augmented = augment_batch(batch, torch.Generator().manual_seed(42))

        unique_masks = torch.unique(augmented.reshape(8, -1), dim=0)
        self.assertGreater(len(unique_masks), 1)

    def test_mask_translation_zeros_wrapped_pixels(self) -> None:
        import torch

        mask = torch.zeros((1, 4, 4))
        mask[:, 0, 0] = 1

        translated = _translate_mask(mask, shift_y=-1, shift_x=-1)

        self.assertEqual(float(translated.sum()), 0.0)


class PcaSvrMaskTest(unittest.TestCase):
    def test_fits_and_predicts_from_binary_mask_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            rows = []
            for index in range(6):
                pixels = np.zeros((8, 8), dtype=np.uint8)
                pixels[2:6, 1 : index + 2] = 255
                Image.fromarray(pixels).save(masks_dir / f"mask-{index}.png")
                rows.append({"file_name": f"mask-{index}", "weight": str(100 + index * 20)})
            model = PcaSvrMaskRegressor(
                masks_dir,
                {"image_size": 8, "n_components": 2, "random_state": 42, "c": 10.0},
            )

            model.fit(rows)
            predictions = model.predict(rows)

        self.assertEqual(predictions.shape, (6,))
        self.assertTrue(np.isfinite(predictions).all())


class PcaFeatureFusionTest(unittest.TestCase):
    def test_fits_geometry_and_mask_components_inside_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            rows = []
            for index in range(8):
                pixels = np.zeros((8, 8), dtype=np.uint8)
                pixels[1:7, 1 : index + 1] = 255
                Image.fromarray(pixels).save(masks_dir / f"mask-{index}.png")
                rows.append(
                    {"file_name": f"mask-{index}", "weight": str(100 + index * 10), "area": str(index + 1)}
                )
            params = {"image_size": 8, "n_components": 3, "n_estimators": 10, "random_state": 42}
            model = PcaFeatureFusionRegressor(masks_dir, params)

            model.fit(rows, ["area"])
            predictions = model.predict(rows, ["area"])

        self.assertEqual(predictions.shape, (8,))
        self.assertTrue(np.isfinite(predictions).all())


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
                "image_size": 8,
                "architecture": "resnet18",
                "estimator": "ridge",
                "n_components": 2,
                "random_state": 42,
            }
            network = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
            model = PretrainedMaskEmbeddingRegressor(masks_dir, params, "cpu")

            with patch("buffalo_weight.pretrained_mask_embedding.build_embedding_network", return_value=network):
                model.fit(rows)
                predictions = model.predict(rows)

        self.assertEqual(predictions.shape, (8,))


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
