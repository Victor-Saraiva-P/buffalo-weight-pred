from __future__ import annotations

from pathlib import Path

from buffalo_weight.report_environment import (
    APPROVED_DEPENDENCIES,
    ComputeEnvironment,
    EnvironmentProvenance,
    PythonRuntime,
    SetupServices,
    WeightSetupStatus,
)


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


def setup_services(
    packages: FakePackageGateway,
    weights: FakeWeightGateway | None = None,
    runtime: FakeRuntimeProbe | None = None,
    writer: RecordingProvenanceWriter | None = None,
) -> SetupServices:
    """Compose named fake adapters; for example, CLI tests can replace one boundary."""
    return SetupServices(
        runtime or FakeRuntimeProbe(),
        packages,
        weights or FakeWeightGateway(),
        writer or RecordingProvenanceWriter(),
    )
