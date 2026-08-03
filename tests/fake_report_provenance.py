from __future__ import annotations

from pathlib import Path


class FixedReportProvenance:
    def inputs_recipe_hash(self) -> str:
        """Return fixed knowledge.

        Example: this isolates recipe discovery in a test.
        """
        return "1" * 64

    def dependencies(self) -> dict[str, str]:
        """Return fixed packages.

        Example: this isolates environment discovery.
        """
        return {"fake-compute": "1.0"}

    def repository_commit(self) -> str:
        """Return fixed source identity.

        Example: this isolates the Git process.
        """
        return "2" * 40


class FixedFeatureSelectionProvenance:
    def feature_selection_recipe_hash(self) -> str:
        """Return fixed selection knowledge; for example, tests get stable manifests."""
        recipe_hash = "3" * 64
        return recipe_hash

    def feature_selection_dependencies(self) -> dict[str, str]:
        """Return fixed selection packages; for example, tests avoid environment coupling."""
        dependencies = {"fake-selection": "2.0"}
        return dependencies

    def repository_commit(self) -> str:
        """Return fixed source identity; for example, tests avoid invoking Git."""
        commit = "2" * 40
        return commit


class FixedFeatureSelectionEnvironment:
    """Record deterministic provenance I/O without reading packages, Git or source files."""

    def __init__(self) -> None:
        self.source_names: list[str] = []
        self.package_names: list[str] = []
        self.commit_roots: list[Path] = []

    def read_source(self, path: Path) -> bytes:
        self.source_names.append(path.name)
        source_identity = path.name.encode()
        return source_identity

    def distribution_version(self, name: str) -> str:
        self.package_names.append(name)
        version = f"fixed-{name}"
        return version

    def repository_commit(self, root: Path) -> str:
        self.commit_roots.append(root)
        commit = "4" * 40
        return commit
