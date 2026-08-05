from __future__ import annotations

import hashlib
import tempfile
import unittest
from subprocess import CompletedProcess
from pathlib import Path
from unittest.mock import patch

from buffalo_weight.environment_contract import ComputeEnvironment
from buffalo_weight.resnet_baseline_adapter import TorchCudaAvailability
from buffalo_weight.resnet_baseline_provenance import (
    LocalResNetBaselineEnvironment,
    SystemResNetBaselineProvenance,
)
from buffalo_weight.resnet_baseline_stage import ScientificResNetBaselineRunner
from tests.fake_resnet_baseline import FakeResNetBaselineEnvironment
from tests.fake_setup_services import FakeRuntimeProbe
from tests.test_resnet18_baseline import RecordingResNetAdapter
from tests.test_resnet18_baseline import sample_ids
from tests.test_resnet18_baseline import tiny_evaluation_samples


class ResNetBaselineSystemTest(unittest.TestCase):
    def test_default_runtime_reads_torch_cuda_availability(self) -> None:
        with patch(
            "buffalo_weight.resnet_baseline_adapter.torch.cuda.is_available",
            FakeCudaProbe(),
        ):
            self.assertTrue(TorchCudaAvailability().cuda_available())

    def test_default_runtime_selects_cuda(self) -> None:
        self.assertEqual(TorchCudaAvailability().training_device().type, "cuda")

    def test_local_environment_reads_source_and_package_metadata(self) -> None:
        environment = LocalResNetBaselineEnvironment()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "recipe.py"
            source.write_bytes(b"recipe source")
            self.assertEqual(environment.read_source(source), b"recipe source")
        with patch(
            "buffalo_weight.resnet_baseline_provenance.importlib.metadata.version",
            FakeDistributionVersion(),
        ):
            self.assertEqual(environment.distribution_version("torch"), "fake:torch")

    def test_local_environment_reads_committed_git_source(self) -> None:
        environment = LocalResNetBaselineEnvironment()
        fake_git = FakeGitRun()
        with patch("buffalo_weight.resnet_baseline_provenance.subprocess.run", fake_git):
            commit = environment.repository_commit(Path("/repository"))
            present = environment.committed_source(Path("/repository"), commit, "recipe.py")
            missing = environment.committed_source(Path("/repository"), "0" * 40, "recipe.py")

        self.assertEqual(commit, "d" * 40)
        self.assertEqual(present, b"committed recipe")
        self.assertIsNone(missing)

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
        environment = FakeResNetBaselineEnvironment()
        provenance = SystemResNetBaselineProvenance(environment)

        recipe_hash = provenance.recipe_hash()
        dependencies = provenance.dependency_versions()
        commit = provenance.repository_commit()

        self.assertEqual(len(recipe_hash), 64)
        self.assertEqual(set(dependencies), {
            "numpy", "Pillow", "scikit-learn", "torch", "torchvision",
        })
        self.assertEqual(commit, environment.commit)
        self.assertEqual(provenance.recipe_hash_at_commit(commit), recipe_hash)
        self.assertIsNone(provenance.recipe_hash_at_commit("0" * 40))


class FakeDistributionVersion:
    def __call__(self, name: str) -> str:
        """Return visible metadata; for example, tests prove the package name is passed."""
        return f"fake:{name}"


class FakeCudaProbe:
    def __call__(self) -> bool:
        """Report CUDA deterministically; for example, no host GPU state is consulted."""
        return True


class FakeGitRun:
    def __call__(
        self, command: list[str], *, check: bool = False,
        capture_output: bool = False, text: bool = False,
    ) -> CompletedProcess[str] | CompletedProcess[bytes]:
        """Return Git-like results; for example, only the fixed commit has a blob."""
        if command[-1] == "HEAD":
            return CompletedProcess(command, 0, stdout="d" * 40 + "\n")
        if command[-1].startswith("d" * 40):
            return CompletedProcess(command, 0, stdout=b"committed recipe")
        return CompletedProcess(command, 1, stdout=b"")


if __name__ == "__main__":
    unittest.main()
