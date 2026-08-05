"""Typed contracts and frozen recipe for the compact CNN baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray


CompactCnnArtifactStatus = Literal["absent", "obsolete", "reusable"]
CompactCnnRunStatus = Literal["absent", "obsolete", "reusable", "rebuilt"]


@dataclass(frozen=True)
class CompactCnnRecipe:
    """Declare the frozen recipe; for example, manifests serialize ``as_mapping()``."""

    image_size: int = 224
    input_channels: int = 1
    optimizer: str = "AdamW"
    learning_rate: float = 0.001
    batch_size: int = 16
    weight_decay: float = 0.0001
    loss: str = "L1"
    max_epochs: int = 300
    patience: int = 40
    minimum_improvement_kg: float = 0.1
    gradient_clip: float = 5.0
    horizontal_flip_probability: float = 0.5
    translation_fraction: float = 0.05
    inner_seed: int = 43
    training_seed: int = 44

    def as_mapping(self) -> dict[str, bool | float | int | str]:
        """Return manifest values; for example, the optimizer remains ``AdamW``."""
        serialized = asdict(self)
        typed = cast(dict[str, bool | float | int | str], serialized)
        return typed


COMPACT_CNN_RECIPE = CompactCnnRecipe()


@dataclass(frozen=True)
class CompactCnnTargetScale:
    mean_kg: float
    scale_kg: float

    def standardize(self, targets_kg: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale permitted targets; for example, validation uses selection statistics."""
        standardized = (targets_kg - self.mean_kg) / self.scale_kg
        typed = np.asarray(standardized, dtype=np.float64)
        return typed

    def restore(self, standardized: NDArray[np.float64]) -> NDArray[np.float64]:
        """Restore kilograms; for example, OOF predictions use report units."""
        targets_kg = standardized * self.scale_kg + self.mean_kg
        typed = np.asarray(targets_kg, dtype=np.float64)
        return typed


@dataclass(frozen=True)
class MaskBatch:
    pixels: NDArray[np.float32]
    targets_kg: NDArray[np.float64]
    sample_ids: tuple[str, ...]
    strata: tuple[str, ...]


class CompactCnnPredictor(Protocol):
    def predict_kg(self, batch: MaskBatch) -> NDArray[np.float64]:
        """Predict held-out masks; for example, values are restored to kilograms."""
        # Prediction receives no augmentation hook, keeping reserved masks unchanged.
        ...


class CompactCnnTrainingAdapter(Protocol):
    def select_epoch_count(
        self, selection: MaskBatch, stopping: MaskBatch,
        target_scale: CompactCnnTargetScale, recipe: CompactCnnRecipe,
    ) -> int:
        """Select epochs using only an inner split; for example, return the best epoch."""
        # The stopping batch is evaluation-only and must never enter gradient updates.
        ...

    def fit_epochs(
        self, training: MaskBatch, target_scale: CompactCnnTargetScale,
        epochs: int, recipe: CompactCnnRecipe,
    ) -> CompactCnnPredictor:
        """Refit from scratch; for example, consume all external-train masks."""
        # Implementations recreate their model before consuming the complete batch.
        ...
