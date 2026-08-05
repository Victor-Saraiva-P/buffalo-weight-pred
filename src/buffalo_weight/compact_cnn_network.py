"""Frozen compact-CNN architecture with deterministic spatial pooling."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class CompactCnnNetwork(nn.Module):
    """Frozen compact architecture; for example, it consumes one 224×224 mask channel."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            DeterministicAdaptiveAveragePool4(), nn.Flatten(), nn.Dropout(0.25),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Predict standardized weight; for example, a batch of two returns shape ``(2,)``."""
        predictions = self.layers(masks)
        squeezed = predictions.squeeze(1)
        return cast(torch.Tensor, squeezed)


class DeterministicAdaptiveAveragePool4(nn.Module):
    """Pool to 4×4 deterministically; for example, 56×56 becomes sixteen 14×14 means."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Average equal spatial bins; for example, frozen 224px inputs reach 56px here."""
        height, width = values.shape[-2:]
        if height % 4 or width % 4:
            raise ValueError(
                f"pool input was {(height, width)!r}; expected dimensions divisible by 4"
            )
        # Issue #19 requires deterministic CUDA, whose adaptive-pool backward is unavailable.
        blocked = values.reshape(*values.shape[:-2], 4, height // 4, 4, width // 4)
        return blocked.mean(dim=(3, 5))
