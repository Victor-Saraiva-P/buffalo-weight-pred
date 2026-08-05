from __future__ import annotations

import unittest

from buffalo_weight.baseline_provenance import SystemBaselineProvenance
from tests.fake_baseline_provenance import FixedBaselineEnvironment


class BaselineProvenanceTest(unittest.TestCase):
    def test_random_forest_source_change_does_not_invalidate_reference_recipe(self) -> None:
        original_environment = FixedBaselineEnvironment()
        original = SystemBaselineProvenance(original_environment)
        changed_environment = FixedBaselineEnvironment(
            ("buffalo_weight.feature_baselines", "RandomForestBaseline")
        )
        changed = SystemBaselineProvenance(changed_environment)

        self.assertNotEqual(
            original.baseline_recipe_hash("random_forest_baseline"),
            changed.baseline_recipe_hash("random_forest_baseline"),
        )
        self.assertEqual(
            original.baseline_recipe_hash("training_mean_reference"),
            changed.baseline_recipe_hash("training_mean_reference"),
        )
        self.assertIn(
            ("buffalo_weight.feature_selection_io", "_read_expected_csv"),
            original_environment.source_requests,
        )

    def test_configuration_dependencies_and_git_identity_use_the_environment(self) -> None:
        environment = FixedBaselineEnvironment()
        provenance = SystemBaselineProvenance(environment)

        candidate = provenance.baseline_dependencies("random_forest_baseline")
        reference = provenance.baseline_dependencies("training_mean_reference")

        self.assertEqual(candidate, {
            "numpy": "fixed-numpy", "scikit-learn": "fixed-scikit-learn",
        })
        self.assertEqual(reference, {"numpy": "fixed-numpy"})
        self.assertEqual(provenance.repository_commit(), "8" * 40)


if __name__ == "__main__":
    unittest.main()
