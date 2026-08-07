"""Paired morphological eligibility checking for sensitivity diagnostics.

Contraction and expansion form an inseparable pair: a mask is eligible
for both or neither. The main evaluation retains all 132 masks; the
morphological diagnostic uses only the eligible subset.

Reference: GitHub Issue #26.
"""

from __future__ import annotations

import numpy as np

from buffalo_weight.diagnostic_sensitivity_perturbations import (
    count_four_neighbor_components,
    euclidean_disk,
    has_valid_topology,
    perturb_contraction,
    perturb_expansion,
)
from buffalo_weight.diagnostic_sensitivity_types import MorphologyEligibility


MORPHOLOGY_DISK_RADIUS_CANONICAL = 5
CANONICAL_LONG_SIDE = 1024


def check_morphology_eligibility(
    mask: np.ndarray, file_name: str, canonical_long_side: int = CANONICAL_LONG_SIDE,
) -> MorphologyEligibility:
    """Determine if a mask is eligible for the contraction+expansion pair.

    Example: ``check_morphology_eligibility(mask, "img01")`` returns eligibility.
    """
    original_long_side = max(mask.shape)
    disk = euclidean_disk(MORPHOLOGY_DISK_RADIUS_CANONICAL, canonical_long_side, original_long_side)
    expanded = perturb_expansion(mask, disk)
    if _expansion_exceeds_margin(mask, expanded):
        return MorphologyEligibility(file_name, "rejected", "insufficient_expansion_margin")
    if not has_valid_topology(mask, expanded):
        return MorphologyEligibility(file_name, "rejected", "expansion_topology_violation")
    contracted = perturb_contraction(mask, disk)
    if not has_valid_topology(mask, contracted):
        return MorphologyEligibility(file_name, "rejected", "contraction_topology_violation")
    return MorphologyEligibility(file_name, "eligible", "")


def _expansion_exceeds_margin(original: np.ndarray, expanded: np.ndarray) -> bool:
    """Check whether expansion touches image border where original did not, or reaches edge.

    Example: ``_expansion_exceeds_margin(mask, expanded)`` returns True if margin is insufficient.
    """
    orig_border = _border_mask(original)
    exp_border = _border_mask(expanded)
    return bool(np.any(exp_border & ~orig_border)) or bool(np.any(exp_border))


def _border_mask(arr: np.ndarray) -> np.ndarray:
    """Extract boolean mask of image border pixels."""
    border = np.zeros(arr.shape, dtype=bool)
    border[0, :] = arr[0, :] > 0
    border[-1, :] = arr[-1, :] > 0
    border[:, 0] = arr[:, 0] > 0
    border[:, -1] = arr[:, -1] > 0
    return border


def compute_all_eligibilities(
    masks: dict[str, np.ndarray],
    canonical_long_side: int = CANONICAL_LONG_SIDE,
) -> list[MorphologyEligibility]:
    """Compute eligibility for all masks, sorted by file_name.

    Example: ``compute_all_eligibilities(masks_dict)`` returns eligibility list.
    """
    eligibilities = [
        check_morphology_eligibility(mask, file_name, canonical_long_side)
        for file_name, mask in sorted(masks.items())
    ]
    return eligibilities
