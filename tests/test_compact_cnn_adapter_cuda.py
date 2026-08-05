from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from buffalo_weight.compact_cnn_adapter import CompactCnnAdapter
from buffalo_weight.compact_cnn_augmentation import augment_binary_masks
from buffalo_weight.compact_cnn_network import DeterministicAdaptiveAveragePool4
from buffalo_weight.compact_cnn_types import (
    COMPACT_CNN_RECIPE, CompactCnnTargetScale, MaskBatch,
)


class UnavailableCompactCnnRuntime:
    """Reject CUDA without touching model operations."""

    def __init__(self) -> None:
        """Start without probes; for example, construction performs the first check."""
        self.availability_checks = 0
        return None

    def cuda_available(self) -> bool:
        """Return false; for example, adapter construction must stop immediately."""
        self.availability_checks += 1
        return False


class CompactCnnAdapterCudaTest(unittest.TestCase):
    def test_frozen_recipe_and_architecture_match_the_protocol(self) -> None:
        adapter = CompactCnnAdapter()
        model = adapter.create_model(44)
        convolutions = [layer for layer in model.layers if isinstance(layer, nn.Conv2d)]
        pools = [layer for layer in model.layers if isinstance(layer, nn.MaxPool2d)]
        dropouts = [layer for layer in model.layers if isinstance(layer, nn.Dropout)]
        dense = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
        adaptive = [layer for layer in model.layers
                    if isinstance(layer, DeterministicAdaptiveAveragePool4)]
        self.assertEqual([(layer.in_channels, layer.out_channels, layer.kernel_size)
                          for layer in convolutions],
                         [(1, 16, (5, 5)), (16, 32, (3, 3)), (32, 64, (3, 3))])
        self.assertEqual(len(pools), 2)
        self.assertEqual(len(adaptive), 1)
        self.assertEqual([layer.p for layer in dropouts], [0.25])
        self.assertEqual([(layer.in_features, layer.out_features) for layer in dense],
                         [(1024, 64), (64, 1)])
        self.assertEqual(COMPACT_CNN_RECIPE.as_mapping(), _expected_recipe())

    def test_augmentation_preserves_binary_foreground_without_cutting(self) -> None:
        masks = torch.zeros((16, 1, 224, 224))
        masks[:, :, 20:180, 30:190] = 1.0
        original_count = masks.sum(dim=(1, 2, 3))

        augmented = augment_binary_masks(masks, torch.Generator().manual_seed(44))

        self.assertTrue(torch.equal(augmented.sum(dim=(1, 2, 3)), original_count))
        self.assertEqual(set(augmented.unique().tolist()), {0.0, 1.0})
        for mask in augmented:
            foreground = torch.nonzero(mask[0] > 0, as_tuple=False)
            self.assertLessEqual(abs(int(foreground[:, 0].min()) - 20), 11)
            self.assertLessEqual(abs(int(foreground[:, 1].min()) - 30), 11)

    def test_cuda_step_checkpoint_device_and_seed_reproduce(self) -> None:
        adapter = CompactCnnAdapter()
        first = adapter.contract_probe(seed=44)
        second = adapter.contract_probe(seed=44)
        self.assertEqual(first.device_type, "cuda")
        self.assertGreater(first.loss, 0.0)
        self.assertTrue(first.has_gradients)
        self.assertTrue(first.parameters_updated)
        self.assertEqual(first.predictions, second.predictions)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compact-cnn.pt"
            adapter.save_model(first.model, path)
            restored = adapter.load_model(path)
            self.assertEqual(first.predictions, adapter.probe_predictions(restored))
            self.assertEqual(next(restored.parameters()).device.type, "cuda")

    def test_minimal_epoch_selection_refit_and_inference_stay_on_cuda(self) -> None:
        adapter = CompactCnnAdapter()
        recipe = replace(COMPACT_CNN_RECIPE, max_epochs=1, patience=1, batch_size=2)
        selection = _mask_batch(4)
        stopping = _mask_batch(2, offset=4)
        inner_scale = CompactCnnTargetScale(105.0, 5.0)

        epochs = adapter.select_epoch_count(selection, stopping, inner_scale, recipe)
        predictor = adapter.fit_epochs(selection, inner_scale, epochs, recipe)
        predictions = predictor.predict_kg(stopping)
        repeated = adapter.fit_epochs(selection, inner_scale, epochs, recipe).predict_kg(stopping)

        self.assertEqual(epochs, 1)
        self.assertEqual(predictions.shape, (2,))
        np.testing.assert_array_equal(predictions, repeated)
        self.assertEqual(next(predictor.model.parameters()).device.type, "cuda")

    def test_missing_cuda_fails_before_model_creation(self) -> None:
        runtime = UnavailableCompactCnnRuntime()

        with self.assertRaisesRegex(ValueError, "CUDA availability was false"):
            CompactCnnAdapter(runtime)

        self.assertEqual(runtime.availability_checks, 1)


def _expected_recipe() -> dict[str, bool | float | int | str]:
    return {
        "image_size": 224, "input_channels": 1, "optimizer": "AdamW",
        "learning_rate": 0.001, "batch_size": 16, "weight_decay": 0.0001,
        "loss": "L1", "max_epochs": 300, "patience": 40,
        "minimum_improvement_kg": 0.1, "gradient_clip": 5.0,
        "horizontal_flip_probability": 0.5, "translation_fraction": 0.05,
        "inner_seed": 43, "training_seed": 44,
    }


def _mask_batch(count: int, offset: int = 0) -> MaskBatch:
    pixels: NDArray[np.float32] = np.zeros((count, 1, 224, 224), dtype=np.float32)
    for index in range(count):
        pixels[index, 0, 30:190, 20 + index : 180 + index] = 1.0
    targets = np.asarray([100.0 + (offset + index) * 2 for index in range(count)])
    identifiers = tuple(f"mask-{offset + index}" for index in range(count))
    strata = tuple("B1" if index % 2 == 0 else "B2" for index in range(count))
    return MaskBatch(pixels, targets, identifiers, strata)


if __name__ == "__main__":
    unittest.main()
