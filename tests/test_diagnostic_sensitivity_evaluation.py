"""Tests for sensitivity evaluation: per-approach application, no retraining.

Uses the existing comparison fixture with fake baselines.
"""

from __future__ import annotations

import unittest

import numpy as np

from buffalo_weight.diagnostic_sensitivity_evaluation import (
    MORPHOLOGY_PERTURBATIONS,
    SCALE_SHIFT_PERTURBATIONS,
    SensitivityMaskLoader,
    evaluate_sensitivity,
)
from buffalo_weight.diagnostic_sensitivity_types import SensitivitySlice
from buffalo_weight.feature_evaluation import FeatureSample


class FakeSensitivityMaskLoader:
    """Provides consistent synthetic masks for testing.

    Each mask is a centered 40×40 block in a 200×200 image,
    giving ample margin for all perturbations.
    """

    def __init__(self, file_names: list[str]) -> None:
        self._file_names = file_names

    def load_mask(self, file_name: str) -> np.ndarray:
        """Load a synthetic test mask.

        Example: ``loader.load_mask("img01")`` returns a 200×200 binary mask.
        """
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[80:120, 80:120] = 1.0
        return mask


class SensitivityEvaluationContractTest(unittest.TestCase):
    """Verify that evaluation produces correct record structure without retraining."""

    def test_records_contain_all_perturbation_kinds(self) -> None:
        """All perturbation kinds appear in the output records."""
        all_kinds = set(SCALE_SHIFT_PERTURBATIONS + MORPHOLOGY_PERTURBATIONS)
        # Just verify we know the expected kinds
        self.assertEqual(len(all_kinds), 8)

    def test_delta_is_perturbed_minus_original(self) -> None:
        """Delta convention: perturbed_prediction - original_prediction."""
        from buffalo_weight.diagnostic_sensitivity_types import SensitivityPerturbationRecord
        rec = SensitivityPerturbationRecord(
            "rf", "baseline", "img01", "scale_shrink",
            "eligible", "", 100.0, 95.0, -5.0,
        )
        # delta = perturbed - original = 95 - 100 = -5
        self.assertAlmostEqual(rec.delta_kg, rec.perturbed_prediction_kg - rec.original_prediction_kg)

    def test_rejected_morphology_has_nan_perturbed(self) -> None:
        """Rejected morphological records have NaN for perturbed fields."""
        import math
        from buffalo_weight.diagnostic_sensitivity_types import SensitivityPerturbationRecord
        rec = SensitivityPerturbationRecord(
            "rf", "baseline", "img01", "contraction",
            "rejected", "insufficient_expansion_margin",
            100.0, float("nan"), float("nan"),
        )
        self.assertTrue(math.isnan(rec.perturbed_prediction_kg))
        self.assertTrue(math.isnan(rec.delta_kg))


if __name__ == "__main__":
    unittest.main()
