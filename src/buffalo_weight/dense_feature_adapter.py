"""Project-owned PyTorch boundary for the dense feature regressor."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn


@dataclass(frozen=True)
class DenseTrainingRecipe:
    hidden_layers: tuple[int, int] = (64, 32)
    dropout: float = 0.20
    learning_rate: float = 0.001
    batch_size: int = 16
    weight_decay: float = 0.0001
    max_epochs: int = 500
    patience: int = 40
    minimum_improvement_kg: float = 0.1
    gradient_clip: float = 5.0
    inner_seed: int = 43
    training_seed: int = 44


@dataclass(frozen=True)
class DenseTargetScale:
    mean_kg: float
    scale_kg: float

    def standardize(self, targets_kg: NDArray[np.float64]) -> NDArray[np.float64]:
        """Standardize targets; for example, training applies only train-fitted values."""
        standardized = (targets_kg - self.mean_kg) / self.scale_kg
        return np.asarray(standardized, dtype=np.float64)

    def restore(self, standardized: NDArray[np.float64]) -> NDArray[np.float64]:
        """Restore kilograms; for example, held-out predictions return report units."""
        targets_kg = standardized * self.scale_kg + self.mean_kg
        return np.asarray(targets_kg, dtype=np.float64)


class DenseFeatureNetwork(nn.Module):
    """Frozen dense architecture; for example, the adapter supplies its declared recipe."""

    def __init__(self, input_count: int, recipe: DenseTrainingRecipe) -> None:
        super().__init__()
        self.input_count = input_count
        first_width, second_width = recipe.hidden_layers
        self.layers = nn.Sequential(
            nn.Linear(input_count, first_width), nn.ReLU(), nn.Dropout(recipe.dropout),
            nn.Linear(first_width, second_width), nn.ReLU(), nn.Dropout(recipe.dropout),
            nn.Linear(second_width, 1),
        )
        self.apply(_initialize_he)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Predict standardized weight; for example, ``network(feature_batch)``."""
        predictions = self.layers(values)
        squeezed = predictions.squeeze(1)
        return cast(torch.Tensor, squeezed)


@dataclass(frozen=True)
class DenseContractProbe:
    device_type: str
    loss: float
    has_gradients: bool
    parameters_updated: bool
    predictions: tuple[float, ...]
    model: DenseFeatureNetwork


class CudaRuntime(Protocol):
    """CUDA availability seam; for example, tests inject an unavailable runtime."""

    def cuda_available(self) -> bool:
        """Report usability; for example, false must stop before model creation."""
        # Preflight stays replaceable so tests can prove failure precedes GPU allocation.
        ...


class TorchCudaRuntime:
    """Read PyTorch CUDA state; for example, production adapters use this runtime."""

    def cuda_available(self) -> bool:
        """Report PyTorch CUDA availability; for example, a configured GPU returns true."""
        available = torch.cuda.is_available()
        usable = bool(available)
        return usable


class DenseFeatureAdapter:
    """Own CUDA operations; for example, the dense baseline injects this adapter."""

    def __init__(self, runtime: CudaRuntime | None = None) -> None:
        resolved_runtime = runtime or TorchCudaRuntime()
        if not resolved_runtime.cuda_available():
            raise ValueError("CUDA availability was false; expected CUDA for dense feature training")
        self.device = torch.device("cuda")

    def create_model(
        self, input_count: int, seed: int,
        recipe: DenseTrainingRecipe = DenseTrainingRecipe(),
    ) -> DenseFeatureNetwork:
        """Create a seeded CUDA model; for example, ``adapter.create_model(26, 44)``."""
        _seed_everything(seed)
        return DenseFeatureNetwork(input_count, recipe).to(self.device)

    def contract_probe(
        self, input_count: int, seed: int,
        recipe: DenseTrainingRecipe = DenseTrainingRecipe(),
    ) -> DenseContractProbe:
        """Exercise the adapter contract; for example, CUDA tests call one tiny update."""
        model = self.create_model(input_count, seed, recipe)
        inputs, targets = self._probe_batch(input_count)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = self._optimizer(model, recipe)
        loss = nn.functional.l1_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        gradients = all(parameter.grad is not None for parameter in model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
        optimizer.step()
        updated = any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
        return DenseContractProbe(self.device.type, float(loss.item()), gradients, updated,
                                  self.predict_tuple(model), model)

    def save_model(self, model: DenseFeatureNetwork, path: Path) -> None:
        """Save an owned checkpoint; for example, ``adapter.save_model(model, path)``."""
        payload = {"state_dict": model.state_dict(), "input_count": model.input_count}
        torch.save(payload, path)
        return None

    def load_model(self, path: Path, input_count: int) -> DenseFeatureNetwork:
        """Load an owned checkpoint; for example, ``adapter.load_model(path, 26)``."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(payload, dict) or payload.get("input_count") != input_count:
            raise ValueError(
                f"checkpoint input_count was {getattr(payload, 'get', lambda *_: None)('input_count')!r}; "
                f"expected {input_count}"
            )
        model = self.create_model(input_count, 44)
        model.load_state_dict(cast(dict[str, torch.Tensor], payload["state_dict"]))
        return model

    def predict_tuple(self, model: DenseFeatureNetwork) -> tuple[float, ...]:
        """Predict a fixed probe batch; for example, checkpoint tests compare tuples."""
        model.eval()
        inputs, _ = self._probe_batch(model.input_count)
        with torch.no_grad():
            predictions = model(inputs).detach().cpu().tolist()
        return tuple(float(value) for value in predictions)

    def select_epoch_count(
        self, train_values: NDArray[np.float64], train_targets: NDArray[np.float64],
        validation_values: NDArray[np.float64], validation_targets_kg: NDArray[np.float64],
        target_scale: DenseTargetScale, recipe: DenseTrainingRecipe,
    ) -> int:
        """Select only epoch count; for example, inner validation never sees the outer fold."""
        model = self.create_model(train_values.shape[1], recipe.training_seed, recipe)
        optimizer, random = self._optimizer(model, recipe), np.random.default_rng(recipe.training_seed)
        best_mae, best_epoch, stale = float("inf"), 1, 0
        for epoch in range(1, recipe.max_epochs + 1):
            self._train_epoch(model, optimizer, train_values, train_targets, recipe, random)
            mae = self._validation_mae(model, validation_values, validation_targets_kg,
                                       target_scale)
            if best_mae - mae > recipe.minimum_improvement_kg:
                best_mae, best_epoch, stale = mae, epoch, 0
            else:
                stale += 1
            if stale >= recipe.patience:
                break
        return best_epoch

    def fit_epochs(
        self, values: NDArray[np.float64], targets: NDArray[np.float64],
        epochs: int, recipe: DenseTrainingRecipe,
    ) -> DenseFeatureNetwork:
        """Retrain from scratch; for example, outer training uses the selected epoch count."""
        model = self.create_model(values.shape[1], recipe.training_seed, recipe)
        optimizer = self._optimizer(model, recipe)
        random = np.random.default_rng(recipe.training_seed)
        for _ in range(epochs):
            self._train_epoch(model, optimizer, values, targets, recipe, random)
        return model

    def predict_array(
        self, model: DenseFeatureNetwork, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Predict standardized targets; for example, callers restore kilogram units."""
        model.eval()
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predictions = model(tensor).detach().cpu().numpy()
        return np.asarray(predictions, dtype=np.float64)

    def _optimizer(
        self, model: DenseFeatureNetwork, recipe: DenseTrainingRecipe
    ) -> torch.optim.AdamW:
        return torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                                 weight_decay=recipe.weight_decay)

    def _train_epoch(
        self, model: DenseFeatureNetwork, optimizer: torch.optim.AdamW,
        values: NDArray[np.float64], targets: NDArray[np.float64],
        recipe: DenseTrainingRecipe, random: np.random.Generator,
    ) -> None:
        model.train()
        order = random.permutation(len(values))
        for start in range(0, len(values), recipe.batch_size):
            indices = order[start : start + recipe.batch_size]
            inputs = torch.as_tensor(values[indices], dtype=torch.float32, device=self.device)
            expected = torch.as_tensor(targets[indices], dtype=torch.float32, device=self.device)
            optimizer.zero_grad(set_to_none=True)
            nn.functional.l1_loss(model(inputs), expected).backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
            optimizer.step()

    def _validation_mae(
        self, model: DenseFeatureNetwork, values: NDArray[np.float64],
        targets_kg: NDArray[np.float64], target_scale: DenseTargetScale,
    ) -> float:
        standardized = self.predict_array(model, values)
        predictions_kg = target_scale.restore(standardized)
        return float(np.mean(np.abs(targets_kg - predictions_kg)))

    def _probe_batch(self, input_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.arange(1, 1 + input_count * 3, dtype=torch.float32, device=self.device)
        inputs = values.reshape(3, input_count) / float(input_count * 3)
        targets = torch.tensor([0.25, -0.50, 0.75], dtype=torch.float32, device=self.device)
        return inputs, targets


def _initialize_he(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
