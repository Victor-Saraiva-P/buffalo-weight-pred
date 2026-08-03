from __future__ import annotations

import unittest

from buffalo_weight.feature_selection_provenance import SystemFeatureSelectionProvenance
from tests.fake_report_provenance import FixedFeatureSelectionEnvironment


class FeatureSelectionProvenanceTest(unittest.TestCase):
    def test_discovers_recipe_dependencies_and_commit_through_owned_boundary(self) -> None:
        environment = FixedFeatureSelectionEnvironment()
        provenance = SystemFeatureSelectionProvenance(environment)

        first_hash = provenance.feature_selection_recipe_hash()
        second_hash = provenance.feature_selection_recipe_hash()
        dependencies = provenance.feature_selection_dependencies()
        commit = provenance.repository_commit()

        self.assertEqual(first_hash, second_hash)
        self.assertEqual(len(first_hash), 64)
        self.assertIn("feature_selection_stage.py", environment.source_names)
        self.assertEqual(set(dependencies), {"numpy", "scipy", "scikit-learn", "matplotlib", "torch"})
        self.assertEqual(commit, "4" * 40)
        self.assertEqual(len(environment.commit_roots), 1)


if __name__ == "__main__":
    unittest.main()
