from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.environment_contract import ComputeEnvironment
from buffalo_weight.resnet_baseline_provenance import SystemResNetBaselineProvenance
from buffalo_weight.resnet_baseline_stage import ScientificResNetBaselineRunner
from tests.fake_setup_services import FakeRuntimeProbe
from tests.test_resnet18_baseline import RecordingResNetAdapter
from tests.test_resnet18_baseline import sample_ids
from tests.test_resnet18_baseline import tiny_evaluation_samples


class ResNetBaselineSystemTest(unittest.TestCase):
    def test_scientific_runner_uses_injected_offline_and_compute_boundaries(self) -> None:
        content = b"verified official weights"
        expected_hash = hashlib.sha256(content).hexdigest()
        runtime = FakeRuntimeProbe(
            compute=ComputeEnvironment("Fake GPU", "7.5", "13.0", "590.1")
        )
        adapter = RecordingResNetAdapter()
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "weights.pth"
            weights_path.write_bytes(content)
            runner = ScientificResNetBaselineRunner(
                adapter, weights_path, expected_hash, runtime
            )

            runner.preflight()
            predictions = runner.evaluate(tiny_evaluation_samples())

        self.assertEqual(len(predictions), 50)
        self.assertEqual({row.file_name for row in predictions}, sample_ids(tiny_evaluation_samples()))
        self.assertEqual(runner.execution_metadata()["gpu_name"], "Fake GPU")

    def test_system_provenance_reads_recipe_dependencies_and_git(self) -> None:
        provenance = SystemResNetBaselineProvenance()

        recipe_hash = provenance.recipe_hash()
        dependencies = provenance.dependency_versions()
        commit = provenance.repository_commit()

        self.assertEqual(len(recipe_hash), 64)
        self.assertEqual(set(dependencies), {
            "numpy", "Pillow", "scikit-learn", "torch", "torchvision",
        })
        self.assertEqual(len(commit), 40)
        self.assertEqual(provenance.recipe_hash_at_commit(commit), recipe_hash)
        self.assertIsNone(provenance.recipe_hash_at_commit("0" * 40))


if __name__ == "__main__":
    unittest.main()
