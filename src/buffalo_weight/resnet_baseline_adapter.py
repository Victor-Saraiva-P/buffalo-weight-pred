"""Project-owned CUDA adapter for the frozen ResNet-18 baseline."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from torch import nn

from buffalo_weight.resnet18_weights import default_offline_resnet18
from buffalo_weight.resnet_baseline_evaluation import (
    ResNetBaselinePredictor,
    ResNetSample,
)


@dataclass(frozen=True)
class ResNetTrainingRecipe:
    image_size: int = 224
    warmup_epochs: int = 20
    warmup_learning_rate: float = 0.001
    layer4_learning_rate: float = 0.0001
    head_learning_rate: float = 0.0005
    batch_size: int = 16
    weight_decay: float = 0.0001
    max_partial_epochs: int = 150
    patience: int = 25
    minimum_improvement_kg: float = 0.1
    gradient_clip: float = 5.0
    inner_seed: int = 43
    training_seed: int = 44
    horizontal_flip_probability: float = 0.5
    translation_fraction: float = 0.05


RESNET18_BASELINE_RECIPE = ResNetTrainingRecipe()


@dataclass(frozen=True)
class _TargetScale:
    mean_kg: float
    standard_deviation_kg: float

    def standardize(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(
            (values - self.mean_kg) / self.standard_deviation_kg, dtype=np.float64
        )

    def restore(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(
            values * self.standard_deviation_kg + self.mean_kg, dtype=np.float64
        )


class CudaAvailability(Protocol):
    def cuda_available(self) -> bool:
        """Report CUDA usability; for example, false stops before model creation."""
        ...


class TorchCudaAvailability:
    def cuda_available(self) -> bool:
        """Read PyTorch CUDA state; for example, production requires an available GPU."""
        return bool(torch.cuda.is_available())


@dataclass(frozen=True)
class ResNetContractProbe:
    device_type: str
    loss: float
    has_gradients: bool
    parameters_updated: bool
    predictions: tuple[float, ...]
    model: ResNet18MaskNetwork


class ResNet18MaskNetwork(nn.Module):
    """Mask-only ResNet-18; for example, the wrapper owns normalization and phases."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        output_layer = cast(nn.Linear, backbone.fc)
        backbone.fc = nn.Linear(output_layer.in_features, 1)
        self.backbone = backbone
        self.register_buffer("image_mean", _channel_values((0.485, 0.456, 0.406)))
        self.register_buffer("image_std", _channel_values((0.229, 0.224, 0.225)))
        self.prepare_head_warmup()

    def normalize_inputs(self, masks: torch.Tensor) -> torch.Tensor:
        """Repeat and normalize masks; for example, one binary channel becomes ImageNet RGB."""
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError(
                f"mask tensor shape was {tuple(masks.shape)!r}; expected N x 1 x H x W"
            )
        repeated = masks.repeat(1, 3, 1, 1)
        return (repeated - self.image_mean) / self.image_std

    def prepare_head_warmup(self) -> None:
        """Freeze the backbone; for example, warm-up updates only the 512→1 head."""
        _set_requires_grad(self.backbone, False)
        _set_requires_grad(self.backbone.fc, True)
        self._phase = "head"

    def prepare_partial_fit(self) -> None:
        """Unfreeze layer4 and head; for example, earlier BatchNorm remains frozen."""
        _set_requires_grad(self.backbone, False)
        _set_requires_grad(self.backbone.layer4, True)
        _set_requires_grad(self.backbone.fc, True)
        self._phase = "partial"

    def train(self, mode: bool = True) -> ResNet18MaskNetwork:
        super().train(mode)
        if not mode:
            return self
        _freeze_earlier_batch_norm(self.backbone, self.backbone.layer4)
        if self._phase == "head":
            self.backbone.layer4.eval()
        return self

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Predict standardized weight; for example, output has one value per mask."""
        predictions = self.backbone(self.normalize_inputs(masks))
        return cast(torch.Tensor, predictions.squeeze(1))


class _CudaResNetPredictor:
    def __init__(
        self, model: ResNet18MaskNetwork, scale: _TargetScale,
        device: torch.device, recipe: ResNetTrainingRecipe,
    ) -> None:
        self._model = model
        self._scale = scale
        self._device = device
        self._recipe = recipe

    def predict(self, samples: tuple[ResNetSample, ...]) -> NDArray[np.float64]:
        """Predict kilograms; for example, inference applies no augmentation."""
        masks = _load_sample_masks(samples, self._recipe.image_size).to(self._device)
        self._model.eval()
        with torch.no_grad():
            values = self._model(masks).detach().cpu().numpy()
        return self._scale.restore(np.asarray(values, dtype=np.float64))


class ResNet18BaselineAdapter:
    """Train the approved two-phase recipe; for example, each refit reloads V1 weights."""

    def __init__(
        self, recipe: ResNetTrainingRecipe = RESNET18_BASELINE_RECIPE,
        network_builder: Callable[[], nn.Module] = default_offline_resnet18,
        runtime: CudaAvailability | None = None,
    ) -> None:
        self.recipe = recipe
        self._network_builder = network_builder
        if not (runtime or TorchCudaAvailability()).cuda_available():
            raise ValueError("CUDA availability was false; expected CUDA for ResNet-18 training")
        self._device = torch.device("cuda")

    def contract_probe(self) -> ResNetContractProbe:
        """Exercise one CUDA update; for example, tests verify gradients and determinism."""
        model = self._new_model()
        model.prepare_partial_fit()
        inputs, targets = _probe_batch(self.recipe.image_size, self._device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        before = [parameter.detach().clone() for parameter in trainable]
        optimizer = _partial_optimizer(model, self.recipe)
        loss = nn.functional.l1_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        has_gradients = all(parameter.grad is not None for parameter in trainable)
        nn.utils.clip_grad_norm_(trainable, self.recipe.gradient_clip)
        optimizer.step()
        updated = any(not torch.equal(old, new) for old, new in zip(before, trainable))
        return ResNetContractProbe(self._device.type, float(loss.item()), has_gradients,
                                   updated, _probe_predictions(model, inputs), model)

    def save_model(self, model: ResNet18MaskNetwork, path: Path) -> None:
        """Save an owned checkpoint; for example, CUDA tests persist the adapter state."""
        torch.save({"state_dict": model.state_dict()}, path)

    def load_model(self, path: Path) -> ResNet18MaskNetwork:
        """Reload an owned checkpoint; for example, no torchvision URL is consulted."""
        payload = torch.load(path, map_location=self._device, weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            raise ValueError(
                f"ResNet checkpoint payload was {type(payload).__name__!r}; "
                "expected a mapping containing state_dict"
            )
        model = self._new_model()
        model.load_state_dict(cast(dict[str, torch.Tensor], payload["state_dict"]))
        return model

    def predict_probe(self, model: ResNet18MaskNetwork) -> tuple[float, ...]:
        """Predict a fixed CUDA batch; for example, checkpoint tests compare values."""
        inputs, _ = _probe_batch(self.recipe.image_size, self._device)
        return _probe_predictions(model, inputs)

    def select_epoch_count(
        self, training: tuple[ResNetSample, ...], validation: tuple[ResNetSample, ...]
    ) -> int:
        """Choose only partial-fit epochs; for example, validation never includes outer rows."""
        scale = _fit_target_scale(training)
        model = self._new_model()
        self._warm_up(model, training, scale)
        model.prepare_partial_fit()
        optimizer = _partial_optimizer(model, self.recipe)
        return self._select_partial_epochs(model, optimizer, training, validation, scale)

    def fit_epochs(
        self, training: tuple[ResNetSample, ...], partial_epochs: int
    ) -> ResNetBaselinePredictor:
        """Refit from V1 weights; for example, warm-up repeats on all outer-train rows."""
        scale = _fit_target_scale(training)
        model = self._new_model()
        self._warm_up(model, training, scale)
        model.prepare_partial_fit()
        optimizer = _partial_optimizer(model, self.recipe)
        self._train_epochs(model, optimizer, training, scale, partial_epochs)
        return _CudaResNetPredictor(model, scale, self._device, self.recipe)

    def _new_model(self) -> ResNet18MaskNetwork:
        _seed_everything(self.recipe.training_seed)
        model = ResNet18MaskNetwork(self._network_builder())
        return model.to(self._device)

    def _warm_up(
        self, model: ResNet18MaskNetwork, samples: tuple[ResNetSample, ...], scale: _TargetScale
    ) -> None:
        model.prepare_head_warmup()
        optimizer = torch.optim.AdamW(
            model.backbone.fc.parameters(), lr=self.recipe.warmup_learning_rate,
            weight_decay=self.recipe.weight_decay,
        )
        self._train_epochs(model, optimizer, samples, scale, self.recipe.warmup_epochs)

    def _train_epochs(
        self, model: ResNet18MaskNetwork, optimizer: torch.optim.AdamW,
        samples: tuple[ResNetSample, ...], scale: _TargetScale, epochs: int,
    ) -> None:
        masks, targets = _training_tensors(samples, scale, self.recipe.image_size)
        generator = torch.Generator().manual_seed(self.recipe.training_seed)
        for _ in range(epochs):
            _train_epoch(model, optimizer, masks, targets, generator, self.recipe, self._device)

    def _select_partial_epochs(
        self, model: ResNet18MaskNetwork, optimizer: torch.optim.AdamW,
        training: tuple[ResNetSample, ...], validation: tuple[ResNetSample, ...],
        scale: _TargetScale,
    ) -> int:
        train_masks, train_targets = _training_tensors(training, scale, self.recipe.image_size)
        generator = torch.Generator().manual_seed(self.recipe.training_seed)
        best_mae, best_epoch, stale = float("inf"), 1, 0
        for epoch in range(1, self.recipe.max_partial_epochs + 1):
            _train_epoch(model, optimizer, train_masks, train_targets, generator,
                         self.recipe, self._device)
            mae = _validation_mae(model, validation, scale, self.recipe, self._device)
            best_mae, best_epoch, stale = _observe_epoch(
                mae, epoch, best_mae, best_epoch, stale, self.recipe.minimum_improvement_kg
            )
            if stale >= self.recipe.patience:
                break
        return best_epoch


def load_letterboxed_mask(path: Path, image_size: int = 224) -> torch.Tensor:
    """Load a binary mask; for example, a wide image receives centered zero padding."""
    with Image.open(path) as source:
        image = source.convert("L")
        _validate_binary_pixels(image, path)
        resized = _letterbox(image, image_size)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(values[None])


def augment_binary_mask(
    mask: torch.Tensor, generator: torch.Generator,
    flip_probability: float, translation_fraction: float,
) -> torch.Tensor:
    """Apply approved safe augmentation; for example, foreground pixels never leave frame."""
    augmented = mask.clone()
    if float(torch.rand((), generator=generator)) < flip_probability:
        augmented = torch.flip(augmented, dims=(2,))
    shift_y, shift_x = _safe_random_shift(augmented, generator, translation_fraction)
    return _translated_mask(augmented, shift_y, shift_x)


def _channel_values(values: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 3, 1, 1)


def _set_requires_grad(module: nn.Module, required: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = required


def _freeze_earlier_batch_norm(backbone: nn.Module, layer4: nn.Module) -> None:
    layer4_modules = set(layer4.modules())
    for module in backbone.modules():
        if isinstance(module, nn.BatchNorm2d) and module not in layer4_modules:
            module.eval()


def _validate_binary_pixels(image: Image.Image, path: Path) -> None:
    unique = set(int(value) for value in np.unique(np.asarray(image)))
    if unique <= {0, 255}:
        return
    raise ValueError(f"mask pixels were {sorted(unique)!r} at {path}; expected only 0 and 255")


def _letterbox(image: Image.Image, image_size: int) -> Image.Image:
    scale = min(image_size / image.width, image_size / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(size, Image.Resampling.NEAREST)
    output = Image.new("L", (image_size, image_size), 0)
    output.paste(resized, ((image_size - size[0]) // 2, (image_size - size[1]) // 2))
    return output


def _safe_random_shift(
    mask: torch.Tensor, generator: torch.Generator, fraction: float
) -> tuple[int, int]:
    foreground = torch.nonzero(mask[0] > 0, as_tuple=False)
    if len(foreground) == 0:
        return 0, 0
    height, width = mask.shape[1:]
    limit_y, limit_x = round(height * fraction), round(width * fraction)
    y_min, x_min = (int(value) for value in foreground.min(dim=0).values)
    y_max, x_max = (int(value) for value in foreground.max(dim=0).values)
    return (_random_between(-min(limit_y, y_min), min(limit_y, height - y_max - 1), generator),
            _random_between(-min(limit_x, x_min), min(limit_x, width - x_max - 1), generator))


def _random_between(low: int, high: int, generator: torch.Generator) -> int:
    return int(torch.randint(low, high + 1, (), generator=generator).item())


def _translated_mask(mask: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    translated = torch.roll(mask, shifts=(shift_y, shift_x), dims=(1, 2))
    if shift_y > 0:
        translated[:, :shift_y] = 0
    elif shift_y < 0:
        translated[:, shift_y:] = 0
    if shift_x > 0:
        translated[:, :, :shift_x] = 0
    elif shift_x < 0:
        translated[:, :, shift_x:] = 0
    return translated


def _fit_target_scale(samples: tuple[ResNetSample, ...]) -> _TargetScale:
    targets = np.asarray([sample.weight_kg for sample in samples], dtype=np.float64)
    deviation = float(np.std(targets))
    return _TargetScale(float(np.mean(targets)), deviation if deviation else 1.0)


def _load_sample_masks(samples: tuple[ResNetSample, ...], size: int) -> torch.Tensor:
    return torch.stack([load_letterboxed_mask(sample.mask_path, size) for sample in samples])


def _training_tensors(
    samples: tuple[ResNetSample, ...], scale: _TargetScale, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    masks = _load_sample_masks(samples, size)
    targets = np.asarray([sample.weight_kg for sample in samples], dtype=np.float64)
    standardized = scale.standardize(targets).astype(np.float32)
    return masks, torch.from_numpy(standardized)


def _partial_optimizer(
    model: ResNet18MaskNetwork, recipe: ResNetTrainingRecipe
) -> torch.optim.AdamW:
    groups = [
        {"params": model.backbone.layer4.parameters(), "lr": recipe.layer4_learning_rate},
        {"params": model.backbone.fc.parameters(), "lr": recipe.head_learning_rate},
    ]
    return torch.optim.AdamW(groups, weight_decay=recipe.weight_decay)


def _train_epoch(
    model: ResNet18MaskNetwork, optimizer: torch.optim.AdamW,
    masks: torch.Tensor, targets: torch.Tensor, generator: torch.Generator,
    recipe: ResNetTrainingRecipe, device: torch.device,
) -> None:
    model.train()
    for indices in _epoch_batches(len(masks), recipe.batch_size, generator):
        batch = torch.stack([
            augment_binary_mask(masks[int(index)], generator,
                                recipe.horizontal_flip_probability,
                                recipe.translation_fraction) for index in indices
        ]).to(device)
        expected = targets[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        nn.functional.l1_loss(model(batch), expected).backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
        optimizer.step()


def _epoch_batches(
    count: int, batch_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    order = torch.randperm(count, generator=generator)
    return tuple(order[start : start + batch_size] for start in range(0, count, batch_size))


def _validation_mae(
    model: ResNet18MaskNetwork, samples: tuple[ResNetSample, ...], scale: _TargetScale,
    recipe: ResNetTrainingRecipe, device: torch.device,
) -> float:
    predictor = _CudaResNetPredictor(model, scale, device, recipe)
    predictions = predictor.predict(samples)
    expected = np.asarray([sample.weight_kg for sample in samples], dtype=np.float64)
    return float(np.mean(np.abs(predictions - expected)))


def _observe_epoch(
    mae: float, epoch: int, best_mae: float, best_epoch: int,
    stale: int, minimum_improvement: float,
) -> tuple[float, int, int]:
    if best_mae - mae > minimum_improvement:
        return mae, epoch, 0
    return best_mae, best_epoch, stale + 1


def _probe_batch(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros((2, 1, size, size), dtype=torch.float32, device=device)
    values[0, :, size // 4 : size // 2, size // 4 : size // 2] = 1
    values[1, :, size // 2 : 3 * size // 4, size // 2 : 3 * size // 4] = 1
    targets = torch.tensor([-0.25, 0.75], dtype=torch.float32, device=device)
    return values, targets


def _probe_predictions(
    model: ResNet18MaskNetwork, inputs: torch.Tensor
) -> tuple[float, ...]:
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).detach().cpu().tolist()
    return tuple(float(value) for value in predictions)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
