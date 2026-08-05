"""Selective source and dependency identity for baseline comparison."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Protocol


class BaselineComparisonProvenance(Protocol):
    """Comparison provenance seam; for example, tests provide fixed identities."""

    def comparison_recipe_hash(self) -> str:
        """Hash pertinent source.

        Example: a plot change invalidates the package.
        """
        ...

    def comparison_dependencies(self) -> dict[str, str]:
        """Return pertinent packages.

        Example: Matplotlib affects PNG outputs.
        """
        ...

    def repository_commit(self) -> str:
        """Return the producing commit.

        Example: manifests retain audit context.
        """
        ...


class BaselineComparisonEnvironment(Protocol):
    """External provenance boundary; for example, tests replace filesystem and Git I/O."""

    def read_source(self, path: Path) -> bytes:
        """Read pertinent source.

        Example: recipe hashing reads each comparison module.
        """
        ...

    def distribution_version(self, name: str) -> str:
        """Read one installed version.

        Example: the package manifest records Matplotlib.
        """
        ...

    def repository_commit(self, root: Path) -> str:
        """Resolve one repository commit.

        Example: provisional evidence records the producing HEAD.
        """
        ...


class BaselineComparisonSourceReader(Protocol):
    """Source-byte seam; for example, local-adapter tests avoid real files."""

    def __call__(self, path: Path) -> bytes:
        """Read one path.

        Example: return fixed recipe bytes.
        """
        ...


class BaselineComparisonVersionReader(Protocol):
    """Package-metadata seam; for example, tests return fixed versions."""

    def __call__(self, name: str) -> str:
        """Read one version.

        Example: resolve the requested distribution.
        """
        ...


class BaselineComparisonGitRunner(Protocol):
    """Git-command seam; for example, tests return a fixed completed process."""

    def __call__(
        self, command: list[str], *, check: bool,
        capture_output: bool, text: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run one Git command.

        Example: resolve repository HEAD.
        """
        ...


class LocalBaselineComparisonEnvironment:
    """Provide local provenance I/O; for example, comparison remains offline."""

    def __init__(
        self, source_reader: BaselineComparisonSourceReader | None = None,
        version_reader: BaselineComparisonVersionReader | None = None,
        git_runner: BaselineComparisonGitRunner | None = None,
    ) -> None:
        """Inject local I/O; for example, tests supply named deterministic fakes."""
        self._source_reader = source_reader or Path.read_bytes
        self._version_reader = version_reader or importlib.metadata.version
        self._git_runner = git_runner or subprocess.run

    def read_source(self, path: Path) -> bytes:
        """Read source bytes; for example, recipe hashes detect implementation edits."""
        source_bytes = self._source_reader(path)
        return source_bytes

    def distribution_version(self, name: str) -> str:
        """Read package metadata; for example, rendering dependencies stay auditable."""
        installed_version = self._version_reader(name)
        return installed_version

    def repository_commit(self, root: Path) -> str:
        """Resolve HEAD; for example, manifests retain the producing checkout."""
        result = self._git_runner(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()


class SystemBaselineComparisonProvenance:
    """Compose comparison provenance; for example, tests inject a named fake environment."""

    def __init__(self, environment: BaselineComparisonEnvironment | None = None) -> None:
        """Inject external I/O; for example, production defaults to local discovery."""
        resolved_environment = environment or LocalBaselineComparisonEnvironment()
        self._environment = resolved_environment

    def comparison_recipe_hash(self) -> str:
        """Hash comparison modules; for example, unrelated training edits are excluded."""
        source_root = Path(__file__).parent
        paths = sorted(source_root.glob("baseline_comparison_*.py"))
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.name.encode())
            digest.update(self._environment.read_source(path))
        return digest.hexdigest()

    def comparison_dependencies(self) -> dict[str, str]:
        """Return rendering dependencies; for example, figure versions are auditable."""
        names = ("numpy", "matplotlib", "Pillow")
        return {name: self._environment.distribution_version(name) for name in names}

    def repository_commit(self) -> str:
        """Return HEAD; for example, the provisional manifest records a full SHA."""
        repository_root = Path(__file__).parents[2]
        commit = self._environment.repository_commit(repository_root)
        return commit
