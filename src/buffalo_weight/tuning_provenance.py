"""Selective source and dependency identity for configuration tuning."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Protocol


class TuningProvenance(Protocol):
    """Tuning provenance seam; for example, tests provide fixed identities."""

    def tuning_recipe_hash(self) -> str:
        """Hash pertinent source.

        Example: a tuning adapter change invalidates the stage.
        """
        ...

    def tuning_dependencies(self) -> dict[str, str]:
        """Return pertinent packages.

        Example: PyTorch or Scikit-learn version is recorded.
        """
        ...

    def repository_commit(self) -> str:
        """Return the producing commit.

        Example: manifests retain audit context.
        """
        ...


class TuningEnvironment(Protocol):
    """External provenance boundary; for example, tests replace filesystem and Git I/O."""

    def read_source(self, path: Path) -> bytes:
        """Read pertinent source.

        Example: recipe hashing reads each tuning module.
        """
        ...

    def distribution_version(self, name: str) -> str:
        """Read one installed version.

        Example: the package manifest records torch.
        """
        ...

    def repository_commit(self, root: Path) -> str:
        """Resolve one repository commit.

        Example: evidence records the producing HEAD.
        """
        ...


class LocalTuningEnvironment:
    """Provide local provenance I/O; for example, tuning remains offline."""

    def __init__(
        self, source_reader: Protocol | None = None,
        version_reader: Protocol | None = None,
        git_runner: Protocol | None = None,
    ) -> None:
        """Inject local I/O; for example, tests supply named deterministic fakes."""
        self._source_reader = source_reader or Path.read_bytes
        self._version_reader = version_reader or importlib.metadata.version
        self._git_runner = git_runner or subprocess.run

    def read_source(self, path: Path) -> bytes:
        """Read source bytes; for example, recipe hashes detect implementation edits."""
        source_bytes: bytes = self._source_reader(path)
        return source_bytes

    def distribution_version(self, name: str) -> str:
        """Read package metadata; for example, dependencies stay auditable."""
        installed_version: str = self._version_reader(name)
        return installed_version

    def repository_commit(self, root: Path) -> str:
        """Resolve HEAD; for example, manifests retain the producing checkout."""
        result = self._git_runner(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()


class SystemTuningProvenance:
    """Compose tuning provenance; for example, tests inject a named fake environment."""

    def __init__(self, environment: TuningEnvironment | None = None) -> None:
        """Inject external I/O; for example, production defaults to local discovery."""
        resolved_environment = environment or LocalTuningEnvironment()
        self._environment = resolved_environment

    def tuning_recipe_hash(self) -> str:
        """Hash tuning modules; for example, unrelated docs edits are excluded."""
        source_root = Path(__file__).parent
        paths = sorted(source_root.glob("tuning_*.py"))
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.name.encode())
            digest.update(self._environment.read_source(path))
        return digest.hexdigest()

    def tuning_dependencies(self) -> dict[str, str]:
        """Return modeling dependencies; for example, sklearn/torch versions are recorded."""
        names = ("numpy", "scikit-learn", "torch")
        return {name: self._environment.distribution_version(name) for name in names}

    def repository_commit(self) -> str:
        """Return HEAD; for example, the manifest records a full SHA."""
        repository_root = Path(__file__).parents[2]
        commit = self._environment.repository_commit(repository_root)
        return commit
