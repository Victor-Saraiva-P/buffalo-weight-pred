"""Project-owned CUDA boundary for the frozen compact CNN baseline."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn


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
        return cast(dict[str, bool | float | int | str], asdict(self))


COMPACT_CNN_RECIPE = CompactCnnRecipe()


@dataclass(frozen=True)
class CompactCnnTargetScale:
    mean_kg: float
    scale_kg: float

    def standardize(self, targets_kg: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale permitted targets; for example, validation uses selection statistics."""
        return np.asarray((targets_kg - self.mean_kg) / self.scale_kg, dtype=np.float64)

    def restore(self, standardized: NDArray[np.float64]) -> NDArray[np.float64]:
        """Restore kilograms; for example, OOF predictions use report units."""
        return np.asarray(standardized * self.scale_kg + self.mean_kg, dtype=np.float64)


@dataclass(frozen=True)
class MaskBatch:
    pixels: NDArray[np.float32]
    targets_kg: NDArray[np.float64]
    sample_ids: tuple[str, ...]
    strata: tuple[str, ...]


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
        return cast(torch.Tensor, self.layers(masks).squeeze(1))


class DeterministicAdaptiveAveragePool4(nn.Module):
    """Pool to 4×4 deterministically; for example, 56×56 becomes sixteen 14×14 means."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Average equal spatial bins; for example, frozen 224px inputs reach 56px here."""
        height, width = values.shape[-2:]
        if height % 4 or width % 4:
            raise ValueError(f"pool input was {(height, width)!r}; expected dimensions divisible by 4")
        # Issue #19 requires deterministic CUDA, whose adaptive-pool backward is unavailable.
        blocked = values.reshape(*values.shape[:-2], 4, height // 4, 4, width // 4)
        return blocked.mean(dim=(3, 5))


class CompactCnnPredictor(Protocol):
    def predict_kg(self, batch: MaskBatch) -> NDArray[np.float64]:
        """Predict held-out masks; for example, values are restored to kilograms."""
        ...


class CompactCnnTrainingAdapter(Protocol):
    def select_epoch_count(
        self, selection: MaskBatch, stopping: MaskBatch,
        target_scale: CompactCnnTargetScale, recipe: CompactCnnRecipe,
    ) -> int:
        """Select epochs using only an inner split; for example, return the best epoch."""
        ...

    def fit_epochs(
        self, training: MaskBatch, target_scale: CompactCnnTargetScale,
        epochs: int, recipe: CompactCnnRecipe,
    ) -> CompactCnnPredictor:
        """Refit from scratch; for example, consume all external-train masks."""
        ...


class CudaRuntime(Protocol):
    def cuda_available(self) -> bool:
        """Report usability; for example, false stops before model creation."""
        ...


class TorchCudaRuntime:
    """Read PyTorch CUDA state; for example, production creates this runtime."""

    def cuda_available(self) -> bool:
        """Report PyTorch CUDA availability; for example, a configured GPU returns true."""
        return bool(torch.cuda.is_available())


@dataclass(frozen=True)
class CompactCnnContractProbe:
    device_type: str
    loss: float
    has_gradients: bool
    parameters_updated: bool
    predictions: tuple[float, ...]
    model: CompactCnnNetwork


class TorchCompactCnnPredictor:
    """Own CUDA inference and target restoration for one refitted fold."""

    def __init__(
        self, model: CompactCnnNetwork, target_scale: CompactCnnTargetScale,
        device: torch.device,
    ) -> None:
        self.model, self.target_scale, self.device = model, target_scale, device

    def predict_kg(self, batch: MaskBatch) -> NDArray[np.float64]:
        """Predict without augmentation; for example, held-out pixels remain unchanged."""
        self.model.eval()
        inputs = torch.as_tensor(batch.pixels, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            standardized = self.model(inputs).detach().cpu().numpy()
        return self.target_scale.restore(np.asarray(standardized, dtype=np.float64))


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
        return CompactCnnNetwork().to(self.device)

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
        return TorchCompactCnnPredictor(model, target_scale, self.device)

    def contract_probe(self, seed: int = 44) -> CompactCnnContractProbe:
        """Exercise CUDA behavior; for example, tests perform one tiny parameter update."""
        model = self.create_model(seed)
        inputs = torch.zeros((2, 1, 32, 32), device=self.device)
        inputs[:, :, 8:24, 6:26] = 1.0
        targets = torch.tensor([-0.5, 0.5], device=self.device)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss = nn.functional.l1_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        gradients = all(parameter.grad is not None for parameter in model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        updated = any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
        predictions = _probe_predictions(model, inputs)
        return CompactCnnContractProbe("cuda", float(loss.item()), gradients, updated,
                                       predictions, model)

    def save_model(self, model: CompactCnnNetwork, path: Path) -> None:
        """Save an owned checkpoint; for example, CUDA tests reload the same weights."""
        torch.save({"state_dict": model.state_dict()}, path)

    def load_model(self, path: Path, seed: int = 44) -> CompactCnnNetwork:
        """Load an owned checkpoint; for example, tensors remain on CUDA."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            raise ValueError(f"checkpoint payload was {type(payload).__name__}; expected state_dict")
        model = self.create_model(seed)
        model.load_state_dict(cast(dict[str, torch.Tensor], payload["state_dict"]))
        return model

    def probe_predictions(self, model: CompactCnnNetwork) -> tuple[float, ...]:
        """Predict the fixed CUDA probe; for example, checkpoint tests compare values."""
        inputs = torch.zeros((2, 1, 32, 32), device=self.device)
        inputs[:, :, 8:24, 6:26] = 1.0
        return _probe_predictions(model, inputs)


def augment_binary_masks(
    masks: torch.Tensor, generator: torch.Generator,
    recipe: CompactCnnRecipe = COMPACT_CNN_RECIPE,
) -> torch.Tensor:
    """Apply only valid flips/translations; for example, foreground pixel count is preserved."""
    augmented = [_augment_one(mask, generator, recipe) for mask in masks]
    return torch.stack(augmented)


def _augment_one(
    mask: torch.Tensor, generator: torch.Generator, recipe: CompactCnnRecipe,
) -> torch.Tensor:
    transformed = mask.clone()
    if torch.rand((), generator=generator) < recipe.horizontal_flip_probability:
        transformed = torch.flip(transformed, dims=[2])
    shift_y, shift_x = _valid_translation(transformed, generator, recipe.translation_fraction)
    return _translate_without_wrap(transformed, shift_y, shift_x)


def _valid_translation(
    mask: torch.Tensor, generator: torch.Generator, fraction: float,
) -> tuple[int, int]:
    foreground = torch.nonzero(mask[0] > 0, as_tuple=False)
    if foreground.numel() == 0:
        return (0, 0)
    height, width = mask.shape[1:]
    y_limits = _translation_limits(foreground[:, 0], height, fraction)
    x_limits = _translation_limits(foreground[:, 1], width, fraction)
    return (_sample_shift(y_limits, generator), _sample_shift(x_limits, generator))


def _translation_limits(
    coordinates: torch.Tensor, size: int, fraction: float,
) -> tuple[int, int]:
    maximum = int(size * fraction)
    lower = max(-maximum, -int(coordinates.min().item()))
    upper = min(maximum, size - 1 - int(coordinates.max().item()))
    return lower, upper


def _sample_shift(limits: tuple[int, int], generator: torch.Generator) -> int:
    lower, upper = limits
    return int(torch.randint(lower, upper + 1, (), generator=generator).item())


def _translate_without_wrap(mask: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    translated = torch.roll(mask, shifts=(shift_y, shift_x), dims=(1, 2))
    if shift_y > 0:
        translated[:, :shift_y, :] = 0
    elif shift_y < 0:
        translated[:, shift_y:, :] = 0
    if shift_x > 0:
        translated[:, :, :shift_x] = 0
    elif shift_x < 0:
        translated[:, :, shift_x:] = 0
    return translated


def _optimizer(
    model: CompactCnnNetwork, recipe: CompactCnnRecipe,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                             weight_decay=recipe.weight_decay)


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
    return float(np.mean(np.abs(batch.targets_kg - predictions)))


def _probe_predictions(
    model: CompactCnnNetwork, inputs: torch.Tensor,
) -> tuple[float, ...]:
    model.eval()
    with torch.no_grad():
        values = model(inputs).detach().cpu().tolist()
    return tuple(float(value) for value in values)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
