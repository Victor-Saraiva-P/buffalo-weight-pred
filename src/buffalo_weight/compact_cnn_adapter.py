"""Project-owned deterministic CUDA training boundary for the compact CNN."""

from __future__ import annotations

import random
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from buffalo_weight.compact_cnn_augmentation import augment_binary_masks
from buffalo_weight.compact_cnn_network import CompactCnnNetwork
from buffalo_weight.compact_cnn_types import (
    CompactCnnRecipe,
    CompactCnnTargetScale,
    MaskBatch,
)


class CudaRuntime(Protocol):
    def cuda_available(self) -> bool:
        """Report usability; for example, false stops before model creation."""
        # Runtime probing remains injectable so CPU-only tests fail before allocation.
        ...


class TorchCudaRuntime:
    """Read PyTorch CUDA state; for example, production creates this runtime."""

    def cuda_available(self) -> bool:
        """Report PyTorch CUDA availability; for example, a configured GPU returns true."""
        available = torch.cuda.is_available()
        usable = bool(available)
        return usable


class TorchCompactCnnPredictor:
    """Own CUDA inference and target restoration for one refitted fold."""

    def __init__(
        self, model: CompactCnnNetwork, target_scale: CompactCnnTargetScale,
        device: torch.device,
    ) -> None:
        self.model = model
        self.target_scale = target_scale
        self.device = device

    def predict_kg(self, batch: MaskBatch) -> NDArray[np.float64]:
        """Predict without augmentation; for example, held-out pixels remain unchanged."""
        self.model.eval()
        inputs = torch.as_tensor(batch.pixels, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            standardized = self.model(inputs).detach().cpu().numpy()
        typed = np.asarray(standardized, dtype=np.float64)
        return self.target_scale.restore(typed)


class CompactCnnAdapter:
    """Own deterministic CUDA training; for example, the baseline injects this adapter."""

    def __init__(self, runtime: CudaRuntime | None = None) -> None:
        resolved_runtime = runtime or TorchCudaRuntime()
        if not resolved_runtime.cuda_available():
            raise ValueError("CUDA availability was false; expected CUDA for compact CNN training")
        self.device = torch.device("cuda")

    def create_model(self, seed: int) -> CompactCnnNetwork:
        """Create a seeded CUDA network; for example, outer refit uses seed 44."""
        _seed_everything(seed)
        model = CompactCnnNetwork()
        cuda_model = model.to(self.device)
        return cuda_model

    def select_epoch_count(
        self, selection: MaskBatch, stopping: MaskBatch,
        target_scale: CompactCnnTargetScale, recipe: CompactCnnRecipe,
    ) -> int:
        """Choose only epoch count; for example, patience observes validation MAE in kg."""
        model = self.create_model(recipe.training_seed)
        optimizer = _optimizer(model, recipe)
        generator = torch.Generator().manual_seed(recipe.training_seed)
        best_mae, best_epoch, stale = float("inf"), 1, 0
        for epoch in range(1, recipe.max_epochs + 1):
            _train_epoch(model, optimizer, selection, target_scale, recipe, generator)
            mae = _validation_mae(model, stopping, target_scale, self.device)
            if best_mae - mae > recipe.minimum_improvement_kg:
                best_mae, best_epoch, stale = mae, epoch, 0
            else:
                stale += 1
            if stale >= recipe.patience:
                break
        return best_epoch

    def fit_epochs(
        self, training: MaskBatch, target_scale: CompactCnnTargetScale,
        epochs: int, recipe: CompactCnnRecipe,
    ) -> TorchCompactCnnPredictor:
        """Refit from scratch; for example, selected epochs train all external rows."""
        model = self.create_model(recipe.training_seed)
        optimizer = _optimizer(model, recipe)
        generator = torch.Generator().manual_seed(recipe.training_seed)
        for _ in range(epochs):
            _train_epoch(model, optimizer, training, target_scale, recipe, generator)
        predictor = TorchCompactCnnPredictor(model, target_scale, self.device)
        return predictor

def _optimizer(model: CompactCnnNetwork, recipe: CompactCnnRecipe) -> torch.optim.AdamW:
    parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=recipe.learning_rate,
                                  weight_decay=recipe.weight_decay)
    return optimizer


def _train_epoch(
    model: CompactCnnNetwork, optimizer: torch.optim.AdamW, batch: MaskBatch,
    target_scale: CompactCnnTargetScale, recipe: CompactCnnRecipe,
    generator: torch.Generator,
) -> None:
    model.train()
    order = torch.randperm(len(batch.sample_ids), generator=generator)
    scaled = target_scale.standardize(batch.targets_kg)
    for start in range(0, len(order), recipe.batch_size):
        indices = order[start : start + recipe.batch_size]
        device = next(model.parameters()).device
        inputs = torch.as_tensor(batch.pixels[indices], device=device)
        inputs = augment_binary_masks(inputs, generator, recipe)
        expected = torch.as_tensor(scaled[indices], dtype=torch.float32, device=inputs.device)
        optimizer.zero_grad(set_to_none=True)
        nn.functional.l1_loss(model(inputs), expected).backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
        optimizer.step()


def _validation_mae(
    model: CompactCnnNetwork, batch: MaskBatch, target_scale: CompactCnnTargetScale,
    device: torch.device,
) -> float:
    model.eval()
    inputs = torch.as_tensor(batch.pixels, dtype=torch.float32, device=device)
    with torch.no_grad():
        standardized = model(inputs).detach().cpu().numpy()
    predictions = target_scale.restore(np.asarray(standardized, dtype=np.float64))
    absolute_errors = np.abs(batch.targets_kg - predictions)
    return float(np.mean(absolute_errors))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
