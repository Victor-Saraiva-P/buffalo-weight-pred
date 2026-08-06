from __future__ import annotations

import unittest
from pathlib import Path

from buffalo_weight.baseline_comparison_provenance import (
    LocalBaselineComparisonEnvironment,
    SystemBaselineComparisonProvenance,
)
from tests.fake_baseline_comparison import (
    FakeBaselineComparisonEnvironment,
    FakeComparisonGitRunner,
    FakeComparisonSourceReader,
    FakeComparisonVersionReader,
)


class BaselineComparisonProvenanceTest(unittest.TestCase):
    def test_hashes_sources_and_records_packages_and_commit_through_environment(self) -> None:
        environment = FakeBaselineComparisonEnvironment()
        provenance = SystemBaselineComparisonProvenance(environment)

        recipe_hash = provenance.comparison_recipe_hash()
        dependencies = provenance.comparison_dependencies()
        commit = provenance.repository_commit()

        self.assertEqual(len(recipe_hash), 64)
        self.assertIn("baseline_comparison_metrics.py", environment.source_names)
        self.assertEqual(environment.package_names, ["numpy", "matplotlib", "Pillow"])
        self.assertEqual(dependencies["numpy"], "fixed-numpy")
        self.assertEqual(commit, "a" * 40)
        self.assertEqual(len(environment.commit_roots), 1)
        changed = SystemBaselineComparisonProvenance(
            FakeBaselineComparisonEnvironment(b"changed")
        ).comparison_recipe_hash()
        self.assertNotEqual(recipe_hash, changed)

    def test_local_environment_uses_injected_source_package_and_git_io(self) -> None:
        source_reader = FakeComparisonSourceReader()
        version_reader = FakeComparisonVersionReader()
        git_runner = FakeComparisonGitRunner()
        environment = LocalBaselineComparisonEnvironment(
            source_reader, version_reader, git_runner,
        )
        source_path = Path("/fixed/comparison.py")
        repository_root = Path("/fixed/repository")

        self.assertEqual(
            environment.read_source(source_path), b"fixed comparison source"
        )
        self.assertEqual(environment.distribution_version("numpy"), "fixed:numpy")
        self.assertEqual(environment.repository_commit(repository_root), "b" * 40)
        self.assertEqual(source_reader.paths, [source_path])
        self.assertEqual(version_reader.names, ["numpy"])
        self.assertEqual(git_runner.commands[0][2], str(repository_root))


if __name__ == "__main__":
    unittest.main()
