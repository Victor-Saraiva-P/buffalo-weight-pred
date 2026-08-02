from __future__ import annotations

import io
import unittest
from pathlib import Path

from buffalo_weight.report_environment import (
    APPROVED_DEPENDENCIES,
    RESNET18_SHA256,
    ComputeEnvironment,
    EnvironmentProvenance,
    PythonRuntime,
    SetupServices,
    WeightSetupStatus,
)
from buffalo_weight.report_cli import main
from buffalo_weight.train_cnn_mask import main as neural_training_main, train_cnn_mask
from buffalo_weight.train_pipeline import main as training_pipeline_main
from buffalo_weight.train_pipeline import train_pipeline


class FakeRuntimeProbe:
    def __init__(
        self,
        python: PythonRuntime = PythonRuntime(3, 14, 6, "CPython"),
        compute: ComputeEnvironment = ComputeEnvironment(None, None, None, None),
    ) -> None:
        self.python = python
        self.compute = compute

    def python_runtime(self) -> PythonRuntime:
        """Return the configured fake Python runtime."""

        return self.python

    def compute_environment(self) -> ComputeEnvironment:
        """Return the configured fake compute environment."""

        return self.compute

    def platform_description(self) -> str:
        """Return a stable fake platform description."""

        return "Fake Linux"


class FakePackageGateway:
    def __init__(self, installed: dict[str, str]) -> None:
        self.installed = installed
        self.install_calls = 0
        self.consistency_checks = 0

    def installed_direct_versions(self) -> dict[str, str]:
        """Return the fake installed direct versions."""

        return dict(self.installed)

    def install_approved(self, requirements_path: Path) -> None:
        """Record installation and install the approved fake versions."""
        self.install_calls += 1
        self.installed = dict(APPROVED_DEPENDENCIES)

    def verify_consistency(self) -> None:
        """Record one fake consistency check."""

        self.consistency_checks += 1

    def resolved_versions(self) -> dict[str, str]:
        """Return direct and fake transitive versions."""

        return {**self.installed, "typing-extensions": "4.15.0"}


class FakeWeightGateway:
    def __init__(
        self,
        result: WeightSetupStatus = WeightSetupStatus.DOWNLOADED,
        actual_hash: str | None = None,
    ) -> None:
        self.result = result
        self.actual_hash = actual_hash
        self.calls = 0

    def ensure_resnet18_weights(
        self, cache_path: Path, expected_sha256: str
    ) -> WeightSetupStatus:
        self.calls += 1
        if self.actual_hash is not None:
            raise ValueError(
                f"ResNet-18 cache SHA-256 was {self.actual_hash!r}; "
                f"expected {expected_sha256!r}"
            )
        return self.result


class RecordingProvenanceWriter:
    def __init__(self) -> None:
        """Start without a recorded provenance value."""

        self.record: EnvironmentProvenance | None = None

    def write(self, path: Path, provenance: EnvironmentProvenance) -> None:
        """Record provenance without filesystem I/O."""

        self.record = provenance


class ReportSetupCliTest(unittest.TestCase):
    def test_setup_installs_and_reports_approved_environment(self) -> None:
        packages = FakePackageGateway({})
        weights = FakeWeightGateway()
        writer = RecordingProvenanceWriter()

        result, stdout, _ = run_setup(packages, weights, writer)

        self.assertEqual(result, 0)
        self.assertIn("validated Python 3.14.6", stdout)
        self.assertIn("installed 8 approved dependencies", stdout)
        self.assertIn("downloaded ResNet-18 IMAGENET1K_V1 weights", stdout)
        self.assertEqual(packages.consistency_checks, 1)
        self.assertIsNotNone(writer.record)

    def test_setup_rejects_python_outside_314(self) -> None:
        runtime = FakeRuntimeProbe(python=PythonRuntime(3, 13, 9, "CPython"))
        packages = FakePackageGateway(dict(APPROVED_DEPENDENCIES))
        stderr = io.StringIO()

        result = main(
            ["setup"],
            setup_services(packages, runtime=runtime),
            stdout=io.StringIO(),
            stderr=stderr,
        )

        self.assertEqual(result, 1)
        self.assertIn("3.13.9", stderr.getvalue())
        self.assertIn("3.14.x", stderr.getvalue())
        self.assertEqual(packages.consistency_checks, 0)

    def test_setup_records_patch_change_without_invalidating_contract(self) -> None:
        runtime = FakeRuntimeProbe(python=PythonRuntime(3, 14, 9, "CPython"))
        writer = RecordingProvenanceWriter()
        stdout = io.StringIO()

        result = main(
            ["setup"],
            setup_services(
                FakePackageGateway(dict(APPROVED_DEPENDENCIES)), runtime=runtime, writer=writer
            ),
            stdout=stdout,
            stderr=io.StringIO(),
        )

        self.assertEqual(result, 0)
        self.assertIn("validated Python 3.14.9", stdout.getvalue())
        self.assertIsNotNone(writer.record)
        provenance = writer.record
        assert provenance is not None
        self.assertEqual(provenance.informational.python_version, "3.14.9")

    def test_setup_rejects_weight_cache_with_wrong_hash(self) -> None:
        bad_hash = "0" * 64
        weights = FakeWeightGateway(actual_hash=bad_hash)
        stderr = io.StringIO()

        result = main(
            ["setup"],
            setup_services(FakePackageGateway(dict(APPROVED_DEPENDENCIES)), weights=weights),
            stdout=io.StringIO(),
            stderr=stderr,
        )

        self.assertEqual(result, 1)
        self.assertIn(bad_hash, stderr.getvalue())
        self.assertIn(RESNET18_SHA256, stderr.getvalue())

    def test_setup_reuses_valid_dependency_and_weight_caches(self) -> None:
        packages = FakePackageGateway(dict(APPROVED_DEPENDENCIES))
        weights = FakeWeightGateway(result=WeightSetupStatus.REUSED)

        result, stdout, _ = run_setup(packages, weights, RecordingProvenanceWriter())

        self.assertEqual(result, 0)
        self.assertIn("reused 8 approved dependencies", stdout)
        self.assertIn("reused ResNet-18 IMAGENET1K_V1 weights", stdout)
        self.assertEqual(packages.install_calls, 0)

    def test_setup_records_missing_cuda_without_rejecting_cpu_safe_work(self) -> None:
        writer = RecordingProvenanceWriter()

        result, stdout, _ = run_setup(
            FakePackageGateway(dict(APPROVED_DEPENDENCIES)),
            FakeWeightGateway(result=WeightSetupStatus.REUSED),
            writer,
        )

        self.assertEqual(result, 0)
        self.assertIn("CUDA unavailable; setup and dry-run remain available", stdout)
        self.assertIsNotNone(writer.record)
        provenance = writer.record
        assert provenance is not None
        self.assertEqual(provenance.validity.python_series, "3.14")
        self.assertEqual(provenance.informational.python_version, "3.14.6")
        self.assertIsNone(provenance.informational.compute.gpu_name)

    def test_neural_cli_rejects_missing_cuda_before_reading_inputs(self) -> None:
        runtime = FakeRuntimeProbe(
            compute=ComputeEnvironment(None, None, "13.0", "590.00")
        )
        stderr = io.StringIO()

        result = neural_training_main(
            ["--shared-config", "missing.yaml", "--models-config", "missing.yaml"],
            runtime_probe=runtime,
            stderr=stderr,
        )

        self.assertEqual(result, 1)
        self.assertIn("CUDA", stderr.getvalue())
        self.assertIn("available CUDA GPU", stderr.getvalue())

    def test_training_pipeline_rejects_missing_cuda_before_reading_inputs(self) -> None:
        runtime = FakeRuntimeProbe(compute=ComputeEnvironment(None, None, "13.0", "590.00"))
        stderr = io.StringIO()

        result = training_pipeline_main(
            [
                "--shared-config",
                "missing-shared.yaml",
                "--classical-models-config",
                "missing-classical.yaml",
                "--cnn-mask-models-config",
                "missing-cnn.yaml",
            ],
            runtime_probe=runtime,
            stderr=stderr,
        )

        self.assertEqual(result, 1)
        self.assertIn("CUDA", stderr.getvalue())
        self.assertIn("available CUDA GPU", stderr.getvalue())

    def test_programmatic_pipeline_rejects_cuda_before_reading_inputs(self) -> None:
        runtime = FakeRuntimeProbe(compute=ComputeEnvironment(None, None, "13.0", "590.00"))

        with self.assertRaisesRegex(ValueError, "available CUDA GPU"):
            train_pipeline(
                Path("missing-shared.yaml"),
                Path("missing-classical.yaml"),
                Path("missing-cnn.yaml"),
                runtime_probe=runtime,
            )

    def test_programmatic_neural_training_rejects_cuda_before_reading_inputs(self) -> None:
        runtime = FakeRuntimeProbe(compute=ComputeEnvironment(None, None, "13.0", "590.00"))

        with self.assertRaisesRegex(ValueError, "available CUDA GPU"):
            train_cnn_mask(
                Path("missing-shared.yaml"),
                Path("missing-models.yaml"),
                runtime_probe=runtime,
            )

def setup_services(
    packages: FakePackageGateway,
    weights: FakeWeightGateway | None = None,
    runtime: FakeRuntimeProbe | None = None,
    writer: RecordingProvenanceWriter | None = None,
) -> SetupServices:
    return SetupServices(
        runtime or FakeRuntimeProbe(),
        packages,
        weights or FakeWeightGateway(),
        writer or RecordingProvenanceWriter(),
    )


def run_setup(
    packages: FakePackageGateway,
    weights: FakeWeightGateway,
    writer: RecordingProvenanceWriter,
) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = main(
        ["setup"], setup_services(packages, weights, writer=writer), stdout=stdout, stderr=stderr
    )
    return result, stdout.getvalue(), stderr.getvalue()


if __name__ == "__main__":
    unittest.main()
