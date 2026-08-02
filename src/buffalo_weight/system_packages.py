"""Package-management adapter for the official dependency contract."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Protocol

from buffalo_weight.report_environment import APPROVED_DEPENDENCIES


class InstalledDistribution(Protocol):
    metadata: Mapping[str, str]
    version: str


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
VersionLookup = Callable[[str], str]
DistributionProvider = Callable[[], Iterable[InstalledDistribution]]


class PipPackageGateway:
    def __init__(
        self,
        python_executable: str,
        command_runner: CommandRunner,
        version_lookup: VersionLookup,
        distribution_provider: DistributionProvider,
    ) -> None:
        self._python_executable = python_executable
        self._command_runner = command_runner
        self._version_lookup = version_lookup
        self._distribution_provider = distribution_provider

    def installed_direct_versions(self) -> dict[str, str]:
        """Read approved versions; for example, missing packages are omitted."""
        installed: dict[str, str] = {}
        for name in APPROVED_DEPENDENCIES:
            version = self._installed_version(name)
            if version is not None:
                installed[name] = version
        return installed

    def install_approved(self, requirements_path: Path) -> None:
        """Install pinned dependencies; for example, consume ``requirements.txt``."""
        command = [
            self._python_executable,
            "-m",
            "pip",
            "install",
            "--requirement",
            str(requirements_path),
        ]
        result = self._command_runner(command, text=True)
        if result.returncode != 0:
            raise ValueError(
                f"pip install exit code was {result.returncode!r}; "
                f"expected 0 for {requirements_path}"
            )

    def verify_consistency(self) -> None:
        """Run pip's environment check; for example, reject incompatible transitives."""
        command = [self._python_executable, "-m", "pip", "check"]
        result = self._command_runner(command, text=True, capture_output=True)
        if result.returncode == 0:
            return
        details = result.stdout.strip() or result.stderr.strip()
        raise ValueError(f"pip check reported {details!r}; expected no dependency conflicts")

    def resolved_versions(self) -> dict[str, str]:
        """Record all distributions; for example, include platform transitives."""
        pairs = (
            (distribution.metadata["Name"], distribution.version)
            for distribution in self._distribution_provider()
        )
        return dict(sorted(pairs, key=lambda pair: pair[0].lower()))

    def _installed_version(self, distribution_name: str) -> str | None:
        try:
            return self._version_lookup(distribution_name)
        except PackageNotFoundError:
            return None
