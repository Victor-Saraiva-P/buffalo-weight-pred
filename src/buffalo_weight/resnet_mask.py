"""Mask preprocessing, augmentation and ResNet-18 spatial network boundary."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image
import torch
from torch import nn


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
        """Preserve frozen BatchNorm state; for example, ``train()`` leaves layer3 BN frozen."""
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
