from __future__ import annotations

import io
import unittest

from buffalo_weight.environment_contract import (
    APPROVED_DEPENDENCIES,
    RESNET18_SHA256,
    PythonRuntime,
    WeightSetupStatus,
)
from buffalo_weight.report_cli import main
from tests.fake_setup_services import (
    FakePackageGateway,
    FakeRuntimeProbe,
    FakeWeightGateway,
    RecordingProvenanceWriter,
    setup_services,
)


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

