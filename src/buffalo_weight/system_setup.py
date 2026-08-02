"""Composition root for operating-system setup adapters."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import distributions, version
from urllib.request import urlopen

import torch

from buffalo_weight.report_environment import RuntimeProbe, SetupServices
from buffalo_weight.system_packages import PipPackageGateway
from buffalo_weight.system_provenance import JsonProvenanceWriter
from buffalo_weight.system_runtime import NvidiaDriverProbe, SystemRuntimeProbe
from buffalo_weight.system_weights import HttpWeightGateway


def default_runtime_probe() -> RuntimeProbe:
    """Build the real runtime adapter; for example, CLI training audits the current host."""
    driver_probe = NvidiaDriverProbe(subprocess.run)
    return SystemRuntimeProbe(
        sys.version_info,
        platform.python_implementation,
        platform.platform,
        torch,
        driver_probe.version,
    )


def default_setup_services() -> SetupServices:
    """Build real setup adapters; for example, the public CLI uses them outside tests."""
    packages = PipPackageGateway(sys.executable, subprocess.run, version, distributions)
    return SetupServices(
        default_runtime_probe(),
        packages,
        HttpWeightGateway(urlopen),
        JsonProvenanceWriter(),
    )
