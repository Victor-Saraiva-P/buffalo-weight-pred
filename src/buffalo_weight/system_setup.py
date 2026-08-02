"""Composition root for operating-system setup adapters."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import distributions, version
from typing import cast
from urllib.request import urlopen

from buffalo_weight.environment_contract import RuntimeProbe, SetupServices
from buffalo_weight.neural_environment import require_neural_cuda_for_run
from buffalo_weight.system_packages import PipPackageGateway
from buffalo_weight.system_provenance import JsonProvenanceWriter
from buffalo_weight.system_runtime import NvidiaDriverProbe, SystemRuntimeProbe, TorchRuntime
from buffalo_weight.system_weights import HttpWeightGateway


class TorchRuntimeImporter:
    def __call__(self) -> TorchRuntime:
        """Import torch after setup installs it; for example, defer CUDA inspection."""

        import torch

        return cast(TorchRuntime, torch)


def default_runtime_probe() -> RuntimeProbe:
    """Build the real runtime adapter; for example, CLI training audits the current host."""
    driver_probe = NvidiaDriverProbe(subprocess.run)
    return SystemRuntimeProbe(
        sys.version_info,
        platform.python_implementation,
        platform.platform,
        TorchRuntimeImporter(),
        driver_probe.version,
    )


def require_official_neural_runtime(
    dry_run: bool, runtime_probe: RuntimeProbe | None = None
) -> None:
    """Gate neural work; for example, a dry-run remains safe on a CPU-only host."""
    probe = runtime_probe or default_runtime_probe()
    require_neural_cuda_for_run(dry_run, probe.compute_environment)


def default_setup_services() -> SetupServices:
    """Build real setup adapters; for example, the public CLI uses them outside tests."""
    packages = PipPackageGateway(sys.executable, subprocess.run, version, distributions)
    return SetupServices(
        default_runtime_probe(),
        packages,
        HttpWeightGateway(urlopen),
        JsonProvenanceWriter(),
    )
