"""Conservative foreground-preserving augmentation for binary masks."""

from __future__ import annotations

import torch

from buffalo_weight.compact_cnn_types import COMPACT_CNN_RECIPE, CompactCnnRecipe


def augment_binary_masks(
    masks: torch.Tensor, generator: torch.Generator,
    recipe: CompactCnnRecipe = COMPACT_CNN_RECIPE,
) -> torch.Tensor:
    """Apply only valid flips/translations; for example, foreground pixel count is preserved."""
    augmented = [_augment_one(mask, generator, recipe) for mask in masks]
    stacked = torch.stack(augmented)
    return stacked


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
    sampled = torch.randint(lower, upper + 1, (), generator=generator)
    return int(sampled.item())


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
