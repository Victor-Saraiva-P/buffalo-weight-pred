"""Operating-system adapters for the official environment setup."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path
from urllib.request import urlopen

from buffalo_weight.report_environment import (
    APPROVED_DEPENDENCIES,
    EnvironmentProvenance,
    PythonRuntime,
    ComputeEnvironment,
    SetupServices,
    WeightSetupStatus,
)
from buffalo_weight.resnet18_weights import validate_resnet18_sha256


RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


class SystemRuntimeProbe:
    def python_runtime(self) -> PythonRuntime:
        """Read the active interpreter; for example, return the running CPython patch."""
        info = sys.version_info
        return PythonRuntime(info.major, info.minor, info.micro, platform.python_implementation())

    def compute_environment(self) -> ComputeEnvironment:
        """Inspect CUDA without requiring it.

        For example, CPU-only setup returns empty GPU fields.
        """
        import torch

        driver = _nvidia_driver_version()
        if not torch.cuda.is_available():
            return ComputeEnvironment(None, None, torch.version.cuda, driver)
        major, minor = torch.cuda.get_device_capability(0)
        return ComputeEnvironment(
            torch.cuda.get_device_name(0), f"{major}.{minor}", torch.version.cuda, driver
        )

    def platform_description(self) -> str:
        """Describe the host for audit.

        For example, include OS and machine architecture.
        """
        return platform.platform()


class PipPackageGateway:
    def installed_direct_versions(self) -> dict[str, str]:
        """Read approved package versions; for example, missing packages are omitted."""
        return {
            name: installed
            for name in APPROVED_DEPENDENCIES
            if (installed := _installed_version(name)) is not None
        }

    def install_approved(self, requirements_path: Path) -> None:
        """Install the pinned contract; for example, consume the repository requirements file."""
        command = [sys.executable, "-m", "pip", "install", "--requirement", str(requirements_path)]
        result = subprocess.run(command, text=True)
        if result.returncode != 0:
            raise ValueError(
                f"pip install exit code was {result.returncode!r}; "
                f"expected 0 for {requirements_path}"
            )

    def verify_consistency(self) -> None:
        """Run pip's environment check; for example, reject incompatible transitives."""
        command = [sys.executable, "-m", "pip", "check"]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            return
        details = result.stdout.strip() or result.stderr.strip()
        raise ValueError(f"pip check reported {details!r}; expected no dependency conflicts")

    def resolved_versions(self) -> dict[str, str]:
        """Record all resolved distributions; for example, include platform transitives."""
        pairs = (
            (distribution.metadata["Name"], distribution.version)
            for distribution in distributions()
        )
        return dict(sorted(pairs, key=lambda pair: pair[0].lower()))


class HttpWeightGateway:
    def ensure_resnet18_weights(
        self, cache_path: Path, expected_sha256: str
    ) -> WeightSetupStatus:
        """Download once and verify every reuse; for example, a valid cache stays offline."""
        if cache_path.exists():
            validate_resnet18_sha256(cache_path, expected_sha256)
            return WeightSetupStatus.REUSED
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = cache_path.with_suffix(f"{cache_path.suffix}.part")
        _download_weight(partial_path)
        validate_resnet18_sha256(partial_path, expected_sha256)
        partial_path.replace(cache_path)
        return WeightSetupStatus.DOWNLOADED


class JsonProvenanceWriter:
    def write(self, path: Path, provenance: EnvironmentProvenance) -> None:
        """Write audit provenance atomically; for example, replace only a complete JSON record."""
        path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = path.with_suffix(f"{path.suffix}.part")
        partial_path.write_text(json.dumps(asdict(provenance), indent=2, sort_keys=True) + "\n")
        partial_path.replace(path)


def default_setup_services() -> SetupServices:
    """Build real setup adapters; for example, the CLI uses these outside tests."""
    return SetupServices(
        SystemRuntimeProbe(), PipPackageGateway(), HttpWeightGateway(), JsonProvenanceWriter()
    )


def _installed_version(distribution_name: str) -> str | None:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None


def _download_weight(destination: Path) -> None:
    try:
        with urlopen(RESNET18_URL, timeout=60) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"ResNet-18 download failed from {RESNET18_URL!r}; expected an accessible official URL"
        ) from error


def _nvidia_driver_version() -> str | None:
    command = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return versions[0] if result.returncode == 0 and versions else None
