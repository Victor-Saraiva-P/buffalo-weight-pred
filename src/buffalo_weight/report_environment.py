"""Official environment contract for the report reproduction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from buffalo_weight.neural_environment import require_neural_cuda


PYTHON_SERIES = "3.14"
APPROVED_DEPENDENCIES: dict[str, str] = {
    "numpy": "2.5.0",
    "Pillow": "12.2.0",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
    "matplotlib": "3.11.0",
    "PyYAML": "6.0.3",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
}
RESNET18_WEIGHT_NAME = "IMAGENET1K_V1"
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RESNET18_CACHE_PATH = Path("generated/setup/resnet18-IMAGENET1K_V1.pth")


class DependencySetupStatus(Enum):
    INSTALLED = "installed"
    REUSED = "reused"


class WeightSetupStatus(Enum):
    DOWNLOADED = "downloaded"
    REUSED = "reused"


@dataclass(frozen=True)
class PythonRuntime:
    major: int
    minor: int
    patch: int
    implementation: str

    @property
    def full_version(self) -> str:
        """Render the full interpreter version.

        Example: ``PythonRuntime(3, 14, 6, "CPython").full_version == "3.14.6"``.
        """
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class ComputeEnvironment:
    gpu_name: str | None
    cuda_capability: str | None
    cuda_version: str | None
    driver_version: str | None


@dataclass(frozen=True)
class ScientificValidity:
    python_series: str
    direct_dependencies: dict[str, str]
    resnet18_weight_name: str
    resnet18_sha256: str


@dataclass(frozen=True)
class InformationalEnvironment:
    python_version: str
    python_implementation: str
    platform: str
    resolved_dependencies: dict[str, str]
    compute: ComputeEnvironment


@dataclass(frozen=True)
class EnvironmentProvenance:
    validity: ScientificValidity
    informational: InformationalEnvironment


@dataclass(frozen=True)
class SetupPaths:
    requirements: Path = Path("requirements.txt")
    weights_cache: Path = RESNET18_CACHE_PATH
    provenance: Path = Path("generated/setup/environment.json")


class RuntimeProbe(Protocol):
    def python_runtime(self) -> PythonRuntime:
        """Read Python provenance; for example, return the active patch version."""

        ...

    def compute_environment(self) -> ComputeEnvironment:
        """Read accelerator provenance; for example, permit an absent GPU during setup."""

        ...

    def platform_description(self) -> str:
        """Describe the host; for example, include its OS and architecture."""

        ...


class PackageGateway(Protocol):
    def installed_direct_versions(self) -> dict[str, str]:
        """Read direct versions; for example, omit an uninstalled approved package."""

        ...

    def install_approved(self, requirements_path: Path) -> None:
        """Install the contract; for example, consume the pinned requirements file."""

        ...

    def verify_consistency(self) -> None:
        """Reject dependency conflicts; for example, expose a failed pip consistency check."""

        ...

    def resolved_versions(self) -> dict[str, str]:
        """Read audit versions; for example, include platform-specific transitives."""

        ...


class WeightGateway(Protocol):
    def ensure_resnet18_weights(
        self, cache_path: Path, expected_sha256: str
    ) -> WeightSetupStatus:
        """Install or reuse weights; for example, validate an existing local cache."""

        ...


class ProvenanceWriter(Protocol):
    def write(self, path: Path, provenance: EnvironmentProvenance) -> None:
        """Persist an audit record; for example, write validity separately from host detail."""

        ...


@dataclass(frozen=True)
class SetupServices:
    runtime: RuntimeProbe
    packages: PackageGateway
    weights: WeightGateway
    provenance: ProvenanceWriter


def validate_python_runtime(runtime: PythonRuntime) -> None:
    """Reject unsupported Python series; for example, Python 3.13 is rejected."""
    if (runtime.major, runtime.minor) == (3, 14):
        return
    raise ValueError(
        f"Python version was {runtime.full_version!r}; expected the 3.14.x series"
    )


def synchronize_dependencies(
    packages: PackageGateway, requirements: Path
) -> DependencySetupStatus:
    """Install or reuse approved dependencies; for example, a matching environment is reused."""
    installed = packages.installed_direct_versions()
    status = DependencySetupStatus.REUSED
    if installed != APPROVED_DEPENDENCIES:
        packages.install_approved(requirements)
        status = DependencySetupStatus.INSTALLED
    validated = packages.installed_direct_versions()
    if validated != APPROVED_DEPENDENCIES:
        raise ValueError(
            f"direct dependency versions were {validated!r}; expected {APPROVED_DEPENDENCIES!r}"
        )
    packages.verify_consistency()
    return status


def build_environment_provenance(
    services: SetupServices, runtime: PythonRuntime, compute: ComputeEnvironment
) -> EnvironmentProvenance:
    """Separate validity inputs from audit detail; for example, patch stays informational."""
    validity = ScientificValidity(
        PYTHON_SERIES, dict(APPROVED_DEPENDENCIES), RESNET18_WEIGHT_NAME, RESNET18_SHA256
    )
    informational = InformationalEnvironment(
        runtime.full_version,
        runtime.implementation,
        services.runtime.platform_description(),
        services.packages.resolved_versions(),
        compute,
    )
    return EnvironmentProvenance(validity, informational)


def setup_official_environment(
    services: SetupServices, paths: SetupPaths = SetupPaths()
) -> list[str]:
    """Prepare the official environment; for example, ``setup_official_environment(services)``."""
    runtime = services.runtime.python_runtime()
    validate_python_runtime(runtime)
    dependency_status = synchronize_dependencies(services.packages, paths.requirements)
    weight_status = services.weights.ensure_resnet18_weights(paths.weights_cache, RESNET18_SHA256)
    compute = services.runtime.compute_environment()
    provenance = build_environment_provenance(services, runtime, compute)
    services.provenance.write(paths.provenance, provenance)
    return _setup_messages(runtime, dependency_status, weight_status, compute, paths.provenance)


def _setup_messages(
    runtime: PythonRuntime,
    dependency_status: DependencySetupStatus,
    weight_status: WeightSetupStatus,
    compute: ComputeEnvironment,
    provenance_path: Path,
) -> list[str]:
    cuda_message = _cuda_message(compute)
    return [
        f"validated Python {runtime.full_version} (3.14.x required; patch is informational)",
        (
            f"{dependency_status.value} {len(APPROVED_DEPENDENCIES)} approved dependencies "
            "and validated consistency"
        ),
        f"{weight_status.value} ResNet-18 {RESNET18_WEIGHT_NAME} weights with SHA-256",
        cuda_message,
        f"recorded environment provenance at {provenance_path}",
    ]
def _cuda_message(compute: ComputeEnvironment) -> str:
    if compute.gpu_name is None:
        return "CUDA unavailable; setup and dry-run remain available"
    return f"recorded CUDA GPU {compute.gpu_name} with capability {compute.cuda_capability}"
