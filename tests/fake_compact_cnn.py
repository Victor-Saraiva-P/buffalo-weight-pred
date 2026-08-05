from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.compact_cnn_adapter import (
    CompactCnnRecipe,
    CompactCnnTargetScale,
    MaskBatch,
)


class FixedCompactCnnPredictor:
    """Return stable predictions; for example, CLI tests avoid scientific training."""

    def __init__(self, owner: RecordingCompactCnnAdapter) -> None:
        self._owner = owner

    def predict_kg(self, batch: MaskBatch) -> NDArray[np.float64]:
        """Predict by sample identity; for example, every held-out row gets one value."""
        self._owner.prediction_calls.append(batch)
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

    def select_epoch_count(
        self, selection: MaskBatch, stopping: MaskBatch,
        target_scale: CompactCnnTargetScale, recipe: CompactCnnRecipe,
    ) -> int:
        """Record isolated epoch selection; for example, return two epochs."""
        self.selection_calls.append((selection, stopping))
        return 2

    def fit_epochs(
        self, training: MaskBatch, target_scale: CompactCnnTargetScale,
        epochs: int, recipe: CompactCnnRecipe,
    ) -> FixedCompactCnnPredictor:
        """Record full outer refit; for example, each fold retrains from scratch."""
        self.refit_calls.append((training, epochs))
        return FixedCompactCnnPredictor(self)


class FixedCompactCnnProvenance:
    """Supply stable scientific identity without package or Git I/O."""

    def __init__(self, recipe_hash: str = "5" * 64) -> None:
        self.recipe_hash = recipe_hash

    def compact_cnn_recipe_hash(self) -> str:
        """Return recipe knowledge; for example, tests can invalidate it."""
        return self.recipe_hash

    def compact_cnn_dependencies(self) -> dict[str, str]:
        """Return frozen packages; for example, manifests include fake PyTorch."""
        return {"torch": "fixed", "numpy": "fixed", "Pillow": "fixed"}

    def repository_commit(self) -> str:
        """Return a full source identity; for example, manifests avoid invoking Git."""
        return "6" * 40

    def compact_cnn_execution(self) -> dict[str, str]:
        """Return CUDA audit fields; for example, CLI tests stay host-independent."""
        return {
            "device": "cuda", "gpu_name": "Fake CUDA GPU", "cuda_capability": "9.0",
            "cuda_version": "13.0", "driver_version": "600.0", "python_version": "3.14.6",
        }
