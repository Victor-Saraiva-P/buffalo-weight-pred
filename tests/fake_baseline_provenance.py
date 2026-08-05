from __future__ import annotations


class FixedBaselineProvenance:
    """Provide deterministic baseline provenance without package or Git I/O."""

    def __init__(self, random_forest_hash: str = "5" * 64) -> None:
        # Tests vary only RF knowledge so the reference should remain reusable.
        self.random_forest_hash = random_forest_hash

    def baseline_recipe_hash(self, configuration: str) -> str:
        """Return fixed recipe knowledge; for example, tests can invalidate only RF."""
        if configuration == "random_forest_baseline":
            return self.random_forest_hash
        return "6" * 64

    def baseline_dependencies(self, configuration: str) -> dict[str, str]:
        """Return fixed packages; for example, tests avoid environment discovery."""
        if configuration == "random_forest_baseline":
            return {"fake-random-forest": "1.0"}
        return {"python": "fixed"}

    def repository_commit(self) -> str:
        """Return fixed Git identity; for example, manifests remain repeatable."""
        commit = "2" * 40
        return commit
