from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from buffalo_weight.feature_selection_provenance import (
    LocalFeatureSelectionEnvironment,
    SystemFeatureSelectionProvenance,
)
from tests.fake_report_provenance import FixedFeatureSelectionEnvironment


class FeatureSelectionProvenanceTest(unittest.TestCase):
    def test_local_environment_reads_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "recipe.py"
            source.write_bytes(b"frozen recipe")
            environment = LocalFeatureSelectionEnvironment()
            self.assertEqual(environment.read_source(source), b"frozen recipe")

    def test_local_environment_reads_installed_dependency(self) -> None:
        environment = LocalFeatureSelectionEnvironment()
        version = environment.distribution_version("numpy")
        self.assertRegex(version, r"^\d+\.\d+")

    def test_local_environment_reads_repository_commit(self) -> None:
        environment = LocalFeatureSelectionEnvironment()
        root = Path(__file__).parents[1]
        commit = environment.repository_commit(root)
        self.assertRegex(commit, r"^[0-9a-f]{40}$")

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
        self.assertEqual(set(dependencies), {
            "numpy", "scipy", "scikit-learn", "matplotlib", "Pillow", "torch",
        })
        self.assertEqual(commit, "4" * 40)
        self.assertEqual(len(environment.commit_roots), 1)


if __name__ == "__main__":
    unittest.main()
