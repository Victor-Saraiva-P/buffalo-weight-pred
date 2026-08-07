"""Tests for paired morphological eligibility checking.

Covers: paired rejection, margin check, topology check, counts.
"""

from __future__ import annotations

import unittest

import numpy as np

from buffalo_weight.diagnostic_sensitivity_eligibility import (
    check_morphology_eligibility,
    compute_all_eligibilities,
)


class MorphologyEligibilityTest(unittest.TestCase):
    def test_eligible_mask_with_margin(self) -> None:
        """A centered mask with plenty of margin is eligible."""
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[50:150, 50:150] = 1.0
        result = check_morphology_eligibility(mask, "well_centered", canonical_long_side=200)
        self.assertEqual(result.status, "eligible")
        self.assertEqual(result.rejection_reason, "")

    def test_rejected_when_expansion_hits_border(self) -> None:
        """Mask near edge is rejected for insufficient expansion margin."""
        mask = np.zeros((50, 50), dtype=np.float32)
        # Foreground very close to the border — expansion will overflow
        mask[1:49, 1:49] = 1.0
        result = check_morphology_eligibility(mask, "edge_mask", canonical_long_side=50)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.rejection_reason, "insufficient_expansion_margin")

    def test_rejected_when_contraction_splits_topology(self) -> None:
        """Two blobs connected by thin bridge: contraction splits them."""
        mask = np.zeros((200, 200), dtype=np.float32)
        # Two distant blobs with plenty of margin
        mask[20:60, 20:60] = 1.0
        mask[120:160, 120:160] = 1.0
        # Very thin bridge — erosion with disk radius will split
        mask[60, 60:120] = 1.0
        result = check_morphology_eligibility(mask, "bridge_mask", canonical_long_side=200)
        self.assertEqual(result.status, "rejected")
        self.assertIn("topology_violation", result.rejection_reason)

    def test_expansion_margin_priority_over_topology(self) -> None:
        """Expansion margin rejection has priority over topology violation."""
        mask = np.zeros((30, 30), dtype=np.float32)
        # Fill almost everything — expansion will overflow AND topology may break
        mask[1:29, 1:29] = 1.0
        result = check_morphology_eligibility(mask, "full_mask", canonical_long_side=30)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.rejection_reason, "insufficient_expansion_margin")

    def test_compute_all_eligibilities_returns_sorted(self) -> None:
        """compute_all_eligibilities returns results sorted by file_name."""
        masks = {
            "z_mask": np.zeros((100, 100), dtype=np.float32),
            "a_mask": np.zeros((100, 100), dtype=np.float32),
        }
        masks["z_mask"][30:70, 30:70] = 1.0
        masks["a_mask"][30:70, 30:70] = 1.0
        results = compute_all_eligibilities(masks, canonical_long_side=100)
        self.assertEqual(results[0].file_name, "a_mask")
        self.assertEqual(results[1].file_name, "z_mask")

    def test_contraction_expansion_pair_is_inseparable(self) -> None:
        """If expansion is rejected, contraction is also rejected (same mask)."""
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[0:48, 0:48] = 1.0  # close to edge
        result = check_morphology_eligibility(mask, "near_edge", canonical_long_side=50)
        # The pair is inseparable — a single eligibility decision covers both
        self.assertEqual(result.status, "rejected")


if __name__ == "__main__":
    unittest.main()
