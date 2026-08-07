"""Deterministic fake tuning provenance for tests."""

from __future__ import annotations


class FixedTuningProvenance:
    """Fixed fake tuning provenance; for example, tests avoid subprocess calls."""

    def __init__(
        self, commit: str = "e456bac" + "0" * 33,
        recipe_hash: str = "a" * 64,
        dependencies: dict[str, str] | None = None,
    ) -> None:
        self._commit = commit
        self._recipe_hash = recipe_hash
        self._dependencies = dependencies or {
            "numpy": "1.26.4",
            "scikit-learn": "1.4.0",
            "torch": "2.2.0",
        }

    def tuning_recipe_hash(self) -> str:
        return self._recipe_hash

    def tuning_dependencies(self) -> dict[str, str]:
        return self._dependencies

    def repository_commit(self) -> str:
        return self._commit
