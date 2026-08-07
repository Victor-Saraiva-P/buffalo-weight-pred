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
    mask: np.ndarray,
    file_name: str,
    canonical_long_side: int = CANONICAL_LONG_SIDE,
) -> MorphologyEligibility:
    """Determine if a mask is eligible for the contraction+expansion pair.

    Checks expansion margin and topology preservation for both operations.
    Expansion with insufficient margin has priority rejection; topology
    violations (elimination, division, union) reject the pair too.

    Example: ``check_morphology_eligibility(mask, "img01")`` returns eligibility.
    """
    original_long_side = max(mask.shape)
    disk = euclidean_disk(MORPHOLOGY_DISK_RADIUS_CANONICAL, canonical_long_side, original_long_side)

    # Priority 1: expansion margin — check if expansion would overflow the image
    expanded = perturb_expansion(mask, disk)
    if _expansion_exceeds_margin(mask, expanded):
        return MorphologyEligibility(file_name, "rejected", "insufficient_expansion_margin")

    # Priority 2: topology of expansion
    if not has_valid_topology(mask, expanded):
        return MorphologyEligibility(file_name, "rejected", "expansion_topology_violation")

    # Priority 3: contraction topology
    contracted = perturb_contraction(mask, disk)
    if not has_valid_topology(mask, contracted):
        return MorphologyEligibility(file_name, "rejected", "contraction_topology_violation")

    # Priority 4: contraction must not eliminate foreground entirely
    if np.sum(contracted > 0) == 0:
        return MorphologyEligibility(file_name, "rejected", "contraction_eliminated_foreground")

    return MorphologyEligibility(file_name, "eligible", "")


def _expansion_exceeds_margin(original: np.ndarray, expanded: np.ndarray) -> bool:
    """Check whether expansion touches the image border where original does not.

    If the expanded mask has foreground on the border rows/columns where the
    original mask had none, that indicates insufficient margin.

    Example: ``_expansion_exceeds_margin(mask, expanded)`` returns True if margin is insufficient.
    """
    h, w = expanded.shape
    # Check if expansion foreground reaches any edge
    border_expanded = np.zeros(expanded.shape, dtype=bool)
    border_expanded[0, :] = expanded[0, :] > 0
    border_expanded[h - 1, :] = expanded[h - 1, :] > 0
    border_expanded[:, 0] = expanded[:, 0] > 0
    border_expanded[:, w - 1] = expanded[:, w - 1] > 0

    border_original = np.zeros(original.shape, dtype=bool)
    border_original[0, :] = original[0, :] > 0
    border_original[h - 1, :] = original[h - 1, :] > 0
    border_original[:, 0] = original[:, 0] > 0
    border_original[:, w - 1] = original[:, w - 1] > 0

    # Expansion exceeds margin if it has NEW border foreground
    new_border = border_expanded & ~border_original
    return bool(np.any(new_border))


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
