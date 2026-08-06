from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess


class FixedBaselineComparisonProvenance:
    def comparison_recipe_hash(self) -> str:
        """Return fixed comparison knowledge.

        Example: tests avoid source discovery.
        """
        return "8" * 64

    def comparison_dependencies(self) -> dict[str, str]:
        """Return fixed packages.

        Example: figure tests avoid environment coupling.
        """
        return {"fake-comparison": "1.0"}

    def repository_commit(self) -> str:
        """Return a fixed audit commit.

        Example: manifests require a full Git SHA.
        """
        return "9" * 40


class FakeBaselineComparisonEnvironment:
    """Record provenance I/O without reading packages, source contents, or Git."""

    def __init__(self, source_suffix: bytes = b"") -> None:
        """Configure source identity; for example, a suffix simulates a source edit."""
        self.source_suffix = source_suffix
        self.source_names: list[str] = []
        self.package_names: list[str] = []
        self.commit_roots: list[Path] = []

    def read_source(self, path: Path) -> bytes:
        """Return named source bytes; for example, tests avoid filesystem contents."""
        self.source_names.append(path.name)
        source_identity = path.name.encode() + self.source_suffix
        return source_identity

    def distribution_version(self, name: str) -> str:
        """Return a named version; for example, tests avoid package metadata."""
        self.package_names.append(name)
        fixed_version = f"fixed-{name}"
        return fixed_version

    def repository_commit(self, root: Path) -> str:
        """Return a fixed commit; for example, tests avoid invoking Git."""
        self.commit_roots.append(root)
        fixed_commit = "a" * 40
        return fixed_commit


class FakeComparisonSourceReader:
    """Return deterministic source bytes while recording the requested path."""

    def __init__(self, source_bytes: bytes = b"fixed comparison source") -> None:
        """Initialize calls; for example, tests assert the exact requested path."""
        self.source_bytes = source_bytes
        self.paths: list[Path] = []

    def __call__(self, path: Path) -> bytes:
        """Read fake source.

        Example: no filesystem access occurs.
        """
        self.paths.append(path)
        source_bytes = self.source_bytes
        return source_bytes


class FakeComparisonVersionReader:
    """Return deterministic package metadata while recording the distribution."""

    def __init__(self, version_prefix: str = "fixed") -> None:
        """Initialize calls; for example, tests inspect requested packages."""
        self.version_prefix = version_prefix
        self.names: list[str] = []

    def __call__(self, name: str) -> str:
        """Read fake metadata.

        Example: no installed package is consulted.
        """
        self.names.append(name)
        fixed_version = f"{self.version_prefix}:{name}"
        return fixed_version


class FakeComparisonGitRunner:
    """Return a deterministic Git result while recording the command."""

    def __init__(self, commit: str = "b" * 40) -> None:
        """Initialize calls; for example, tests inspect the repository argument."""
        self.commit = commit
        self.commands: list[list[str]] = []

    def __call__(
        self, command: list[str], *, check: bool,
        capture_output: bool, text: bool,
    ) -> CompletedProcess[str]:
        """Run fake Git.

        Example: no subprocess is started.
        """
        self.commands.append(command)
        stdout = self.commit + "\n"
        return CompletedProcess(command, 0, stdout=stdout)
