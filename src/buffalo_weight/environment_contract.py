"""Types and constants that define the official reproducible environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol


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
        """Render the version; for example, Python 3.14.6 renders as ``3.14.6``."""
        version_numbers = (self.major, self.minor, self.patch)
        return ".".join(str(number) for number in version_numbers)


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
        """Read accelerator provenance; for example, permit absent GPU during setup."""

        ...

    def platform_description(self) -> str:
        """Describe the host; for example, include its OS and architecture."""

        ...


class PackageGateway(Protocol):
    def installed_direct_versions(self) -> dict[str, str]:
        """Read approved versions; for example, omit an uninstalled package."""

        ...

    def install_approved(self, requirements_path: Path) -> None:
        """Install the contract; for example, consume the pinned requirements file."""

        ...

    def verify_consistency(self) -> None:
        """Reject dependency conflicts; for example, expose a failed pip check."""

        ...

    def resolved_versions(self) -> dict[str, str]:
        """Read audit versions; for example, include platform-specific transitives."""

        ...


class WeightGateway(Protocol):
    def ensure_resnet18_weights(
        self, cache_path: Path, expected_sha256: str
    ) -> WeightSetupStatus:
        """Install or reuse weights; for example, validate an existing cache."""

        ...


class ProvenanceWriter(Protocol):
    def write(self, path: Path, provenance: EnvironmentProvenance) -> None:
        """Persist an audit record; for example, separate validity from host detail."""

        ...


@dataclass(frozen=True)
class SetupServices:
    runtime: RuntimeProbe
    packages: PackageGateway
    weights: WeightGateway
    provenance: ProvenanceWriter
