"""Mask perturbation primitives for controlled sensitivity analysis.

Perturbations are applied in-memory to binary masks without modifying
the curated Máscaras Binarizadas on disk.

Reference: GitHub Issue #26.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label


def euclidean_disk(radius_canonical: int, canonical_long_side: int, original_long_side: int) -> np.ndarray:
    """Build a Euclidean disk structuring element at canonical scale, mapped to original grid.

    The disk radius is specified in pixels at canonical scale (1024). It is converted
    to the original grid by ``radius_original = round(radius_canonical * original_long_side / canonical_long_side)``,
    with ties rounded up (math.ceil-style half-up).

    Example: ``euclidean_disk(5, 1024, 512)`` produces a disk of radius ~3 in original pixels.
    """
    # Convert radius from canonical to original grid, rounding ties up
    ratio = original_long_side / canonical_long_side
    radius_original = _round_half_up(radius_canonical * ratio)
    radius_original = max(radius_original, 1)
    return _build_disk(radius_original)


def _round_half_up(value: float) -> int:
    """Round to nearest integer, breaking ties toward positive infinity.

    Example: ``_round_half_up(2.5)`` returns 3; ``_round_half_up(2.4)`` returns 2.
    """
    return math.floor(value + 0.5)


def _build_disk(radius: int) -> np.ndarray:
    """Build a binary disk structuring element of given pixel radius.

    Example: ``_build_disk(2)`` returns a 5×5 array with Euclidean disk shape.
    """
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    # Euclidean distance criterion: points within or on the circle boundary
    disk = (x * x + y * y) <= radius * radius
    return disk.astype(np.uint8)


def perturb_scale_shrink(mask: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Shrink foreground by ``fraction`` around its center of mass.

    Example: ``perturb_scale_shrink(mask, 0.05)`` produces 5% smaller foreground.
    """
    return _rescale_foreground(mask, 1.0 - fraction)


def perturb_scale_grow(mask: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Grow foreground by ``fraction`` around its center of mass.

    Example: ``perturb_scale_grow(mask, 0.05)`` produces 5% larger foreground.
    """
    return _rescale_foreground(mask, 1.0 + fraction)


def _rescale_foreground(mask: np.ndarray, scale_factor: float) -> np.ndarray:
    """Rescale foreground pixels around their center of mass, keeping image size fixed.

    Uses inverse mapping (destination-to-source) so growing fills all destination pixels
    without gaps from discrete rounding.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask.copy()
    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    h, w = mask.shape
    # Inverse mapping: for each destination pixel, find source coordinate
    all_ys, all_xs = np.mgrid[0:h, 0:w]
    # Map destination back to source: source = center + (dest - center) / scale
    src_ys = cy + (all_ys.astype(np.float64) - cy) / scale_factor
    src_xs = cx + (all_xs.astype(np.float64) - cx) / scale_factor
    src_ys_int = np.round(src_ys).astype(int)
    src_xs_int = np.round(src_xs).astype(int)
    valid = (src_ys_int >= 0) & (src_ys_int < h) & (src_xs_int >= 0) & (src_xs_int < w)
    result = np.zeros_like(mask)
    result[valid] = mask[src_ys_int[valid], src_xs_int[valid]]
    return result.astype(mask.dtype)


def perturb_shift(mask: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, bool]:
    """Shift foreground by (dy, dx) pixels. Returns (shifted_mask, cuts_foreground).

    ``cuts_foreground`` is True if any foreground pixel would leave the image boundary.

    Example: ``perturb_shift(mask, -5, 0)`` shifts up by 5 pixels.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask.copy(), False
    h, w = mask.shape
    new_ys = ys + dy
    new_xs = xs + dx
    cuts = bool(np.any((new_ys < 0) | (new_ys >= h) | (new_xs < 0) | (new_xs >= w)))
    result = np.zeros_like(mask)
    valid = (new_ys >= 0) & (new_ys < h) & (new_xs >= 0) & (new_xs < w)
    result[new_ys[valid], new_xs[valid]] = 1
    return result.astype(mask.dtype), cuts


def compute_shift_pixels(mask: np.ndarray, fraction: float = 0.05) -> int:
    """Compute shift amount in pixels as fraction of the mask image's long side.

    Example: ``compute_shift_pixels(mask_100x80, 0.05)`` returns 5.
    """
    long_side = max(mask.shape)
    return max(_round_half_up(long_side * fraction), 1)


def perturb_contraction(mask: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Erode (contract) the mask foreground using the given structuring element.

    Example: ``perturb_contraction(mask, disk)`` returns contracted mask.
    """
    return binary_erosion(mask, structure=structuring_element).astype(mask.dtype)


def perturb_expansion(mask: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Dilate (expand) the mask foreground using the given structuring element.

    Example: ``perturb_expansion(mask, disk)`` returns expanded mask.
    """
    return binary_dilation(mask, structure=structuring_element).astype(mask.dtype)


def count_four_neighbor_components(mask: np.ndarray) -> int:
    """Count connected components using 4-neighbor connectivity.

    Example: ``count_four_neighbor_components(mask)`` returns the component count.
    """
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    _, count = label(mask > 0, structure=structure)
    return int(count)


def has_valid_topology(original_mask: np.ndarray, perturbed_mask: np.ndarray) -> bool:
    """Check strict 1:1 component correspondence between original and perturbed masks.

    Neither component elimination, splitting, nor merging is permitted.

    Example: ``has_valid_topology(orig, pert)`` returns True if 1:1 topology holds.
    """
    orig_count = count_four_neighbor_components(original_mask)
    pert_count = count_four_neighbor_components(perturbed_mask)
    if orig_count != pert_count or orig_count == 0:
        return orig_count == pert_count
    return _verify_component_mapping(original_mask, perturbed_mask, orig_count)


def _verify_component_mapping(orig: np.ndarray, pert: np.ndarray, count: int) -> bool:
    """Verify that every original component maps 1:1 to a unique perturbed component."""
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    orig_labeled, _ = label(orig > 0, structure=struct)
    pert_labeled, _ = label(pert > 0, structure=struct)
    mapped_pert_labels: set[int] = set()
    for comp_id in range(1, count + 1):
        overlapping = pert_labeled[orig_labeled == comp_id]
        nonzero_overlaps = overlapping[overlapping > 0]
        if len(nonzero_overlaps) == 0:
            return False
        unique_targets = set(nonzero_overlaps)
        if len(unique_targets) != 1:
            return False
        mapped_pert_labels.add(next(iter(unique_targets)))
    return len(mapped_pert_labels) == count
