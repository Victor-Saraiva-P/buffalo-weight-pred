from __future__ import annotations

from pathlib import Path

from buffalo_weight.baseline_types import BaselineConfiguration


class FixedBaselineEnvironment:
    """Provide replaceable source, package and Git reads for provenance tests."""

    def __init__(self, source_override: tuple[str, str] | None = None) -> None:
        # One override isolates whether only the consuming configuration changes.
        self.source_override = source_override
        self.source_requests: list[tuple[str, str]] = []
        self.package_requests: list[str] = []

    def source_text(self, module_name: str, symbol_name: str) -> str:
        """Return stable source; for example, one RF symbol can be changed selectively."""
        self.source_requests.append((module_name, symbol_name))
        if self.source_override == (module_name, symbol_name):
            return f"changed:{module_name}:{symbol_name}"
        return f"fixed:{module_name}:{symbol_name}"

    def distribution_version(self, name: str) -> str:
        """Return a stable version; for example, scikit-learn resolves without metadata I/O."""
        self.package_requests.append(name)
        return f"fixed-{name}"

    def repository_commit(self, root: Path) -> str:
        """Return stable Git identity; for example, no subprocess runs in unit tests."""
        del root
        return "8" * 40


class FixedBaselineProvenance:
    """Provide deterministic baseline provenance without package or Git I/O."""

    def __init__(self, random_forest_hash: str = "5" * 64) -> None:
        # Tests vary only RF knowledge so the reference should remain reusable.
        resolved_hash = random_forest_hash
        self.random_forest_hash = resolved_hash

    def baseline_recipe_hash(self, configuration: BaselineConfiguration) -> str:
        """Return fixed recipe knowledge; for example, tests can invalidate only RF."""
        if configuration == "random_forest_baseline":
            return self.random_forest_hash
        return "6" * 64

    def baseline_dependencies(
        self, configuration: BaselineConfiguration,
    ) -> dict[str, str]:
        """Return fixed packages; for example, tests avoid environment discovery."""
        if configuration == "random_forest_baseline":
            return {"fake-random-forest": "1.0"}
        return {"python": "fixed"}

    def repository_commit(self) -> str:
        """Return fixed Git identity; for example, manifests remain repeatable."""
        commit = "2" * 40
        return commit
