"""Runtime and accelerator adapters for official environment inspection."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from buffalo_weight.report_environment import ComputeEnvironment, PythonRuntime


class PythonVersionInfo(Protocol):
    major: int
    minor: int
    micro: int


class CudaRuntime(Protocol):
    def is_available(self) -> bool: ...

    def get_device_capability(self, device: int) -> tuple[int, int]: ...

    def get_device_name(self, device: int) -> str: ...


class TorchVersion(Protocol):
    cuda: str | None


class TorchRuntime(Protocol):
    cuda: CudaRuntime
    version: TorchVersion


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class NvidiaDriverProbe:
    _command_runner: CommandRunner

    def version(self) -> str | None:
        """Read the NVIDIA driver version; for example, return ``"590.00"``."""
        command = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        try:
            result = self._command_runner(command, text=True, capture_output=True, check=False)
        except FileNotFoundError:
            return None
        versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return versions[0] if result.returncode == 0 and versions else None


class SystemRuntimeProbe:
    def __init__(
        self,
        version_info: PythonVersionInfo,
        implementation_provider: Callable[[], str],
        platform_provider: Callable[[], str],
        torch_runtime: TorchRuntime,
        driver_version_provider: Callable[[], str | None],
    ) -> None:
        self._version_info = version_info
        self._implementation_provider = implementation_provider
        self._platform_provider = platform_provider
        self._torch_runtime = torch_runtime
        self._driver_version_provider = driver_version_provider

    def python_runtime(self) -> PythonRuntime:
        """Read the active interpreter; for example, return the running CPython patch."""
        info = self._version_info
        return PythonRuntime(
            info.major, info.minor, info.micro, self._implementation_provider()
        )

    def compute_environment(self) -> ComputeEnvironment:
        """Inspect CUDA; for example, CPU-only setup returns empty GPU fields."""
        cuda = self._torch_runtime.cuda
        driver = self._driver_version_provider()
        if not cuda.is_available():
            return ComputeEnvironment(None, None, self._torch_runtime.version.cuda, driver)
        major, minor = cuda.get_device_capability(0)
        return ComputeEnvironment(
            cuda.get_device_name(0),
            f"{major}.{minor}",
            self._torch_runtime.version.cuda,
            driver,
        )

    def platform_description(self) -> str:
        """Describe the host; for example, include OS and machine architecture."""
        return self._platform_provider()
