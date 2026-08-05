from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import torch

from buffalo_weight.compact_cnn_augmentation import augment_binary_masks
from buffalo_weight.compact_cnn_types import (
    CompactCnnRecipe,
    CompactCnnTargetScale,
    MaskBatch,
)


class FixedCompactCnnPredictor:
    """Return stable predictions; for example, CLI tests avoid scientific training."""

    def __init__(self, owner: RecordingCompactCnnAdapter) -> None:
        """Retain the recorder; for example, prediction calls remain observable."""
        self._owner = owner
        return None

    def predict_kg(self, batch: MaskBatch) -> NDArray[np.float64]:
        """Predict by sample identity; for example, every held-out row gets one value."""
        self._owner.prediction_calls.append(batch)
        self._owner.evaluation_pixels.append(batch.pixels.copy())
        values = [
            90.0 + int(sample_id.rsplit("-", maxsplit=1)[-1].split(".", maxsplit=1)[0])
            for sample_id in batch.sample_ids
        ]
        return np.asarray(values, dtype=np.float64)


class RecordingCompactCnnAdapter:
    """Record neural partitions while replacing CUDA work in CLI acceptance tests."""

    def __init__(self) -> None:
        self.selection_calls: list[tuple[MaskBatch, MaskBatch]] = []
        self.refit_calls: list[tuple[MaskBatch, int]] = []
        self.prediction_calls: list[MaskBatch] = []
        self.augmented_training: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []
        self.evaluation_pixels: list[NDArray[np.float32]] = []

    def select_epoch_count(
        self, selection: MaskBatch, stopping: MaskBatch,
        target_scale: CompactCnnTargetScale, recipe: CompactCnnRecipe,
    ) -> int:
        """Record isolated epoch selection; for example, return two epochs."""
        self.selection_calls.append((selection, stopping))
        self._record_augmentation(selection, recipe)
        self.evaluation_pixels.append(stopping.pixels.copy())
        return 2

    def fit_epochs(
        self, training: MaskBatch, target_scale: CompactCnnTargetScale,
        epochs: int, recipe: CompactCnnRecipe,
    ) -> FixedCompactCnnPredictor:
        """Record full outer refit; for example, each fold retrains from scratch."""
        self.refit_calls.append((training, epochs))
        self._record_augmentation(training, recipe)
        return FixedCompactCnnPredictor(self)

    def _record_augmentation(self, batch: MaskBatch, recipe: CompactCnnRecipe) -> None:
        original = batch.pixels.copy()
        tensor = torch.as_tensor(original)
        augmented = augment_binary_masks(tensor, torch.Generator().manual_seed(44), recipe)
        typed = np.asarray(augmented.numpy(), dtype=np.float32)
        self.augmented_training.append((original, typed))


class FailingCompactCnnAdapter:
    """Fail after freshness classification; for example, tests prove fail-closed removal."""

    def select_epoch_count(
        self, selection: MaskBatch, stopping: MaskBatch,
        target_scale: CompactCnnTargetScale, recipe: CompactCnnRecipe,
    ) -> int:
        """Reject training; for example, no replacement artifact can be published."""
        raise ValueError("injected compact CNN training failure")

    def fit_epochs(
        self, training: MaskBatch, target_scale: CompactCnnTargetScale,
        epochs: int, recipe: CompactCnnRecipe,
    ) -> FixedCompactCnnPredictor:
        """Remain unreachable; for example, selection already failed closed."""
        raise ValueError("unexpected compact CNN refit after selection failure")


class FixedCompactCnnProvenance:
    """Supply stable scientific identity without package or Git I/O."""

    def __init__(self, recipe_hash: str = "5" * 64) -> None:
        """Set recipe identity; for example, another hash makes artifacts obsolete."""
        self.recipe_hash = recipe_hash
        return None

    def compact_cnn_recipe_hash(self) -> str:
        """Return recipe knowledge; for example, tests can invalidate it."""
        recipe_hash = self.recipe_hash
        return recipe_hash

    def compact_cnn_dependencies(self) -> dict[str, str]:
        """Return frozen packages; for example, manifests include fake PyTorch."""
        dependencies = {"torch": "fixed", "numpy": "fixed", "Pillow": "fixed"}
        return dependencies

    def repository_commit(self) -> str:
        """Return a full source identity; for example, manifests avoid invoking Git."""
        commit = "6" * 40
        return commit

    def compact_cnn_execution(self) -> dict[str, str]:
        """Return CUDA audit fields; for example, CLI tests stay host-independent."""
        return {
            "device": "cuda", "gpu_name": "Fake CUDA GPU", "cuda_capability": "9.0",
            "cuda_version": "13.0", "driver_version": "600.0", "python_version": "3.14.6",
        }
