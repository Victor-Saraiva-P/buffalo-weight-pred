from __future__ import annotations

import hashlib
import io
import json
import subprocess
import unittest
from pathlib import Path
from typing import cast

from buffalo_weight.report_environment import (
    APPROVED_DEPENDENCIES,
    ComputeEnvironment,
    EnvironmentProvenance,
    InformationalEnvironment,
    ScientificValidity,
    WeightSetupStatus,
)
from buffalo_weight.system_packages import PipPackageGateway
from buffalo_weight.system_provenance import JsonProvenanceWriter
from buffalo_weight.system_runtime import NvidiaDriverProbe, SystemRuntimeProbe
from buffalo_weight.system_setup import default_setup_services
from buffalo_weight.system_weights import HttpWeightGateway
from tests.fake_filesystem import MemoryPath


class FakeUrlOpen:
    def __init__(self, content: bytes) -> None:
        """Prepare one deterministic HTTP response body."""
        self.content = content
        self.requested_url = ""
        self.timeout = 0

    def __call__(self, url: str, timeout: int) -> io.BytesIO:
        """Record the request and return the configured response."""
        self.requested_url = url
        self.timeout = timeout
        return io.BytesIO(self.content)


class RecordingCommandRunner:
    def __init__(self, stdout: str = "") -> None:
        """Prepare successful subprocess results with stable output."""
        self.commands: list[list[str]] = []
        self.stdout = stdout

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Record a subprocess command and return success."""
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=self.stdout, stderr="")


class FakeVersionLookup:
    def __call__(self, distribution_name: str) -> str:
        """Return the approved version for a direct dependency."""
        return APPROVED_DEPENDENCIES[distribution_name]


class FakeDistribution:
    def __init__(self, name: str, installed_version: str) -> None:
        """Expose importlib distribution metadata without package I/O."""
        self.metadata = {"Name": name}
        self.version = installed_version


class FakeDistributionProvider:
    def __call__(self) -> list[FakeDistribution]:
        """Return one deterministic transitive distribution."""
        return [FakeDistribution("typing-extensions", "4.15.0")]


class FakeCudaRuntime:
    def __init__(self, available: bool = False) -> None:
        """Prepare deterministic CUDA state."""
        self.available = available

    def is_available(self) -> bool:
        """Report configured CUDA availability."""
        return self.available

    def get_device_capability(self, device: int) -> tuple[int, int]:
        """Return a fixed capability for the requested fake device."""
        return (9, 0)

    def get_device_name(self, device: int) -> str:
        """Return a fixed name for the requested fake device."""
        return "Fake GPU"


class FakeTorchVersion:
    cuda = "13.0"


class FakeTorchRuntime:
    def __init__(self, available: bool = False) -> None:
        """Expose the torch fields used by the runtime adapter."""
        self.cuda = FakeCudaRuntime(available)
        self.version = FakeTorchVersion()


class FakeVersionInfo:
    major = 3
    minor = 14
    micro = 9


class FixedTextProbe:
    def __init__(self, value: str) -> None:
        """Prepare one deterministic platform text value."""
        self.value = value

    def __call__(self) -> str:
        """Return the configured platform text."""
        return self.value


class SystemSetupTest(unittest.TestCase):
    def test_system_runtime_probe_records_runtime_and_platform(self) -> None:
        probe = SystemRuntimeProbe(
            FakeVersionInfo(),
            FixedTextProbe("CPython"),
            FixedTextProbe("Fake Linux"),
            FakeTorchRuntime(),
            FixedTextProbe("590.00"),
        )

        runtime = probe.python_runtime()
        platform_description = probe.platform_description()

        self.assertEqual(runtime.full_version, "3.14.9")
        self.assertEqual(runtime.implementation, "CPython")
        self.assertEqual(platform_description, "Fake Linux")

    def test_system_runtime_probe_records_missing_cuda_and_driver(self) -> None:
        probe = SystemRuntimeProbe(
            FakeVersionInfo(),
            FixedTextProbe("CPython"),
            FixedTextProbe("Fake Linux"),
            FakeTorchRuntime(),
            FixedTextProbe("590.00"),
        )

        compute = probe.compute_environment()

        self.assertIsNone(compute.gpu_name)
        self.assertEqual(compute.driver_version, "590.00")

    def test_package_gateway_installs_checks_and_records_versions(self) -> None:
        runner = RecordingCommandRunner()
        gateway = PipPackageGateway(
            "/fake/python", runner, FakeVersionLookup(), FakeDistributionProvider()
        )

        installed = gateway.installed_direct_versions()
        gateway.install_approved(Path("requirements.txt"))
        gateway.verify_consistency()
        resolved = gateway.resolved_versions()

        self.assertEqual(installed, APPROVED_DEPENDENCIES)
        self.assertEqual(resolved, {"typing-extensions": "4.15.0"})
        self.assertIn("pip", runner.commands[0])
        self.assertEqual(runner.commands[1][-1], "check")

    def test_nvidia_driver_probe_injects_command_runner(self) -> None:
        runner = RecordingCommandRunner("590.00\n")

        version = NvidiaDriverProbe(runner).version()

        self.assertEqual(version, "590.00")
        self.assertEqual(runner.commands[0][0], "nvidia-smi")

    def test_default_setup_services_uses_system_adapters(self) -> None:
        services = default_setup_services()

        self.assertIsInstance(services.runtime, SystemRuntimeProbe)
        self.assertIsInstance(services.packages, PipPackageGateway)
        self.assertIsInstance(services.weights, HttpWeightGateway)
        self.assertIsInstance(services.provenance, JsonProvenanceWriter)

    def test_weight_gateway_reuses_cache_with_expected_sha256(self) -> None:
        cache_path = MemoryPath("weights.pth", {"weights.pth": b"official weights"})
        expected = hashlib.sha256(b"official weights").hexdigest()

        status = HttpWeightGateway(FakeUrlOpen(b"")).ensure_resnet18_weights(
            cast(Path, cache_path), expected
        )

        self.assertEqual(status, WeightSetupStatus.REUSED)

    def test_weight_gateway_rejects_existing_cache_with_wrong_sha256(self) -> None:
        cache_path = MemoryPath("weights.pth", {"weights.pth": b"corrupt weights"})

        with self.assertRaisesRegex(ValueError, "ResNet-18 cache SHA-256.*expected"):
            HttpWeightGateway(FakeUrlOpen(b"")).ensure_resnet18_weights(
                cast(Path, cache_path), "0" * 64
            )

    def test_weight_gateway_downloads_valid_weight_into_offline_cache(self) -> None:
        content = b"official weights"
        expected = hashlib.sha256(content).hexdigest()
        cache_path = MemoryPath("weights.pth")
        url_open = FakeUrlOpen(content)

        status = HttpWeightGateway(url_open).ensure_resnet18_weights(
            cast(Path, cache_path), expected
        )

        self.assertEqual(status, WeightSetupStatus.DOWNLOADED)
        self.assertEqual(cache_path.files["weights.pth"], content)
        self.assertEqual(url_open.timeout, 60)

    def test_provenance_writer_serializes_validity_and_information_separately(self) -> None:
        provenance = EnvironmentProvenance(
            ScientificValidity("3.14", {"numpy": "2.5.0"}, "IMAGENET1K_V1", "abc"),
            InformationalEnvironment(
                "3.14.6",
                "CPython",
                "Linux",
                {"numpy": "2.5.0", "nvidia-cublas": "13.0"},
                ComputeEnvironment("Fake GPU", "9.0", "13.0", "590.00"),
            ),
        )

        path = MemoryPath("environment.json")
        JsonProvenanceWriter().write(cast(Path, path), provenance)
        record = json.loads(path.files["environment.json"].decode())

        self.assertEqual(record["validity"]["python_series"], "3.14")
        self.assertEqual(record["informational"]["python_version"], "3.14.6")
        self.assertEqual(record["informational"]["compute"]["driver_version"], "590.00")


if __name__ == "__main__":
    unittest.main()
